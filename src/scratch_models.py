from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _stft(waveform: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )


def _istft(spectrum: torch.Tensor, length: int, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=spectrum.device, dtype=spectrum.real.dtype)
    return torch.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=length,
    )


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride=(1, 1)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DCCRNFromScratch(nn.Module):
    """Compact complex-mask convolutional recurrent network trained from random weights."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128, hidden_size: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.encoder1 = ConvBlock(2, 16, stride=(2, 1))
        self.encoder2 = ConvBlock(16, 32, stride=(2, 1))
        self.encoder3 = ConvBlock(32, 64, stride=(2, 1))
        bottleneck_bins = (n_fft // 2 + 1 + 7) // 8
        bottleneck_size = 64 * bottleneck_bins
        self.input_projection = nn.Linear(bottleneck_size, hidden_size)
        self.recurrent = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, bottleneck_size)
        self.decoder2 = ConvBlock(64 + 32, 32)
        self.decoder1 = ConvBlock(32 + 16, 16)
        self.mask = nn.Sequential(nn.Conv2d(16 + 2, 2, 3, padding=1), nn.Tanh())

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        spectrum = _stft(noisy, self.n_fft, self.hop_length)
        features = torch.stack((spectrum.real, spectrum.imag), dim=1)
        encoded1 = self.encoder1(features)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)

        batch, channels, frequencies, frames = encoded3.shape
        sequence = encoded3.permute(0, 3, 1, 2).reshape(batch, frames, -1)
        sequence = self.input_projection(sequence)
        sequence, _ = self.recurrent(sequence)
        sequence = self.output_projection(sequence)
        decoded = sequence.reshape(batch, frames, channels, frequencies).permute(0, 2, 3, 1)

        decoded = F.interpolate(decoded, size=encoded2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder2(torch.cat((decoded, encoded2), dim=1))
        decoded = F.interpolate(decoded, size=encoded1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decoder1(torch.cat((decoded, encoded1), dim=1))
        decoded = F.interpolate(decoded, size=features.shape[-2:], mode="bilinear", align_corners=False)
        complex_mask = self.mask(torch.cat((decoded, features), dim=1))

        mask_real, mask_imag = complex_mask[:, 0], complex_mask[:, 1]
        enhanced_real = mask_real * spectrum.real - mask_imag * spectrum.imag
        enhanced_imag = mask_real * spectrum.imag + mask_imag * spectrum.real
        enhanced_spectrum = torch.complex(enhanced_real, enhanced_imag)
        return _istft(enhanced_spectrum, noisy.shape[-1], self.n_fft, self.hop_length)


class RNNoiseLikeFromScratch(nn.Module):
    """A causal recurrent spectral-mask model inspired by RNNoise, trained from random weights."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, hidden_size: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        bins = n_fft // 2 + 1
        self.input_projection = nn.Sequential(nn.Linear(bins, hidden_size), nn.Tanh())
        self.recurrent = nn.GRU(hidden_size, hidden_size, num_layers=3, batch_first=True)
        self.mask = nn.Sequential(nn.Linear(hidden_size, bins), nn.Sigmoid())

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        spectrum = _stft(noisy, self.n_fft, self.hop_length)
        magnitude = spectrum.abs()
        features = torch.log1p(magnitude).transpose(1, 2)
        hidden = self.input_projection(features)
        hidden, _ = self.recurrent(hidden)
        gain = self.mask(hidden).transpose(1, 2)
        enhanced_spectrum = spectrum * gain
        return _istft(enhanced_spectrum, noisy.shape[-1], self.n_fft, self.hop_length)


class WaveUNetFromScratch(nn.Module):
    """One-dimensional encoder-decoder with waveform skip connections."""

    def __init__(self, base_channels: int = 24, depth: int = 5, kernel_size: int = 15):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        channels = [base_channels * (2**index) for index in range(depth)]
        self.encoders = nn.ModuleList()
        input_channels = 1
        for output_channels in channels:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    nn.GroupNorm(1, output_channels),
                    nn.PReLU(output_channels),
                )
            )
            input_channels = output_channels
        bottleneck_channels = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], bottleneck_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, bottleneck_channels),
            nn.PReLU(bottleneck_channels),
        )
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        current_channels + skip_channels,
                        skip_channels,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.GroupNorm(1, skip_channels),
                    nn.PReLU(skip_channels),
                )
            )
            current_channels = skip_channels
        self.output = nn.Conv1d(channels[0], 1, 1)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        inputs = noisy.unsqueeze(1)
        encoded = inputs
        skips = []
        for encoder in self.encoders:
            encoded = encoder(encoded)
            skips.append(encoded)
        decoded = self.bottleneck(encoded)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            decoded = F.interpolate(decoded, size=skip.shape[-1], mode="linear", align_corners=False)
            decoded = decoder(torch.cat((decoded, skip), dim=1))
        decoded = F.interpolate(decoded, size=inputs.shape[-1], mode="linear", align_corners=False)
        return torch.tanh(self.output(decoded)).squeeze(1)


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "dccrn":
        return DCCRNFromScratch(**kwargs)
    if name == "rnnoise_like":
        return RNNoiseLikeFromScratch(**kwargs)
    if name == "wave_unet":
        return WaveUNetFromScratch(**kwargs)
    raise ValueError(f"Unknown model: {name}")

