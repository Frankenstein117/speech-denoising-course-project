import unittest

import torch

from src.scratch_models import DCCRNFromScratch, RNNoiseLikeFromScratch, WaveUNetFromScratch


class ScratchModelTests(unittest.TestCase):
    def test_dccrn_preserves_waveform_shape_and_backpropagates(self):
        model = DCCRNFromScratch(hidden_size=32)
        waveform = torch.randn(1, 4096)
        output = model(waveform)
        self.assertEqual(output.shape, waveform.shape)
        output.abs().mean().backward()

    def test_rnnoise_like_preserves_waveform_shape_and_backpropagates(self):
        model = RNNoiseLikeFromScratch(hidden_size=32)
        waveform = torch.randn(1, 4096)
        output = model(waveform)
        self.assertEqual(output.shape, waveform.shape)
        output.abs().mean().backward()

    def test_wave_unet_preserves_waveform_shape_and_backpropagates(self):
        model = WaveUNetFromScratch(base_channels=4, depth=3)
        waveform = torch.randn(1, 4096)
        output = model(waveform)
        self.assertEqual(output.shape, waveform.shape)
        output.abs().mean().backward()


if __name__ == "__main__":
    unittest.main()
