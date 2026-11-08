"""The runtime metrics must match scikit-image (used only in development)."""
import numpy as np
import pytest

from app.imaging import metrics

skimage = pytest.importorskip("skimage")


def _imgs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.normal(0, 12, a.shape), 0, 255).astype(np.uint8)
    return a, b


def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity
    a, b = _imgs()
    ga, gb = a[..., 0], b[..., 0]
    assert metrics.ssim(ga, gb) == pytest.approx(structural_similarity(ga, gb, data_range=255), abs=2e-3)


def test_lab_and_ciede2000_match_skimage():
    from skimage.color import deltaE_ciede2000, rgb2lab
    a, b = _imgs(1)
    np.testing.assert_allclose(metrics.rgb2lab(a), rgb2lab(a), atol=1e-3)
    de_ref = deltaE_ciede2000(rgb2lab(a), rgb2lab(b))
    np.testing.assert_allclose(metrics.delta_e_ciede2000(metrics.rgb2lab(a), metrics.rgb2lab(b)), de_ref, atol=1e-3)


def test_ciede2000_reference_pairs():
    # Sharma et al. 2005 test data (first pairs)
    lab1 = np.array([[50.0, 2.6772, -79.7751], [50.0, 3.1571, -77.2803], [50.0, 2.8361, -74.0200]])
    lab2 = np.array([[50.0, 0.0, -82.7485], [50.0, 0.0, -82.7485], [50.0, 0.0, -82.7485]])
    np.testing.assert_allclose(metrics.delta_e_ciede2000(lab1, lab2), [2.0425, 2.8615, 3.4412], atol=1e-4)
