import unittest

import numpy as np

from src.metrics import align_to_reference, segmental_snr_db, snr_db


class MetricTests(unittest.TestCase):
    def test_alignment_removes_positive_delay(self) -> None:
        rng = np.random.default_rng(3)
        reference = rng.normal(size=8000)
        estimate = np.concatenate([np.zeros(160), reference])
        aligned_reference, aligned_estimate, lag = align_to_reference(
            reference, estimate, sample_rate=8000
        )
        self.assertEqual(lag, 160)
        np.testing.assert_allclose(aligned_reference, aligned_estimate)

    def test_snr_improves_when_error_shrinks(self) -> None:
        rng = np.random.default_rng(4)
        reference = rng.normal(size=16000)
        noisy = reference + 0.3 * rng.normal(size=16000)
        enhanced = reference + 0.1 * rng.normal(size=16000)
        self.assertGreater(snr_db(reference, enhanced), snr_db(reference, noisy))
        self.assertGreater(
            segmental_snr_db(reference, enhanced, 16000),
            segmental_snr_db(reference, noisy, 16000),
        )


if __name__ == "__main__":
    unittest.main()

