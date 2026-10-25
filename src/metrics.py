from __future__ import annotations

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy import signal


EPS = np.finfo(np.float64).eps


def as_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = np.gcd(source_rate, target_rate)
    return signal.resample_poly(audio, target_rate // divisor, source_rate // divisor)


def align_to_reference(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int, max_shift_ms: float = 250
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align an estimate to a reference by bounded normalized cross-correlation."""
    reference = as_mono_float(reference)
    estimate = as_mono_float(estimate)
    max_shift = int(sample_rate * max_shift_ms / 1000)
    decimation = max(1, sample_rate // 8000)
    ref_small = reference[::decimation]
    est_small = estimate[::decimation]
    corr = signal.correlate(est_small, ref_small, mode="full", method="fft")
    lags = signal.correlation_lags(len(est_small), len(ref_small), mode="full")
    bound = max_shift // decimation
    mask = np.abs(lags) <= bound
    lag = int(lags[mask][np.argmax(corr[mask])] * decimation)

    if lag > 0:
        estimate = estimate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length], lag


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = as_mono_float(reference)
    estimate = as_mono_float(estimate)
    length = min(len(reference), len(estimate))
    reference, estimate = reference[:length], estimate[:length]
    return float(10 * np.log10((np.sum(reference**2) + EPS) / (np.sum((reference-estimate)**2) + EPS)))


def segmental_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30,
    clip: tuple[float, float] = (-10, 35),
) -> float:
    frame = max(1, int(sample_rate * frame_ms / 1000))
    length = min(len(reference), len(estimate))
    scores: list[float] = []
    for start in range(0, length - frame + 1, frame):
        clean_frame = reference[start : start + frame]
        estimate_frame = estimate[start : start + frame]
        if np.mean(clean_frame**2) < 1e-8:
            continue
        score = 10 * np.log10(
            (np.sum(clean_frame**2) + EPS)
            / (np.sum((clean_frame - estimate_frame) ** 2) + EPS)
        )
        scores.append(float(np.clip(score, *clip)))
    return float(np.mean(scores)) if scores else float("nan")


def full_reference_metrics(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int
) -> dict[str, float]:
    reference, estimate, lag = align_to_reference(reference, estimate, sample_rate)
    ref16 = resample(reference, sample_rate, 16000)
    est16 = resample(estimate, sample_rate, 16000)
    ref8 = resample(reference, sample_rate, 8000)
    est8 = resample(estimate, sample_rate, 8000)
    return {
        "snr_db": snr_db(reference, estimate),
        "ssnr_db": segmental_snr_db(reference, estimate, sample_rate),
        "pesq_wb": float(pesq(16000, ref16, est16, "wb")),
        "pesq_nb": float(pesq(8000, ref8, est8, "nb")),
        "stoi": float(stoi(ref16, est16, 16000, extended=False)),
        "alignment_lag_samples": lag,
    }

