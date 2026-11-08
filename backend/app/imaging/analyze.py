"""Cheap image diagnostics and fidelity checks."""
from __future__ import annotations

import cv2
import numpy as np
from .metrics import delta_e_ciede2000, rgb2lab, ssim as structural_ssim


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _limit(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper). Only comparable at equal size."""
    return float(cv2.Laplacian(_gray(rgb), cv2.CV_64F).var())


def noise_sigma(rgb: np.ndarray) -> float:
    """Immerkær's fast noise variance estimate, in 0-255 grey levels."""
    g = _gray(_limit(rgb, 1024)).astype(np.float64)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def diagnose(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    small = _limit(rgb, 512)
    return {
        "width": w, "height": h,
        "sharpness_512": round(sharpness(small), 1),
        "noise_sigma": round(noise_sigma(rgb), 2),
    }


def fidelity(original: np.ndarray, enhanced: np.ndarray, max_side: int = 768) -> dict:
    """Compare the enhanced image, scaled back to the original's size, with the original.

    Both are lightly blurred first so legitimate denoising / sharpening isn't penalised;
    what remains measures structural changes (hallucinated objects) and colour drift.
    """
    oh, ow = original.shape[:2]
    back = cv2.resize(enhanced, (ow, oh), interpolation=cv2.INTER_AREA)
    a, b = _limit(original, max_side), _limit(back, max_side)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    b = cv2.GaussianBlur(b, (0, 0), 1.0)
    ssim = structural_ssim(_gray(a), _gray(b))
    lab_a = rgb2lab(cv2.GaussianBlur(a, (0, 0), 2.0))
    lab_b = rgb2lab(cv2.GaussianBlur(b, (0, 0), 2.0))
    de = delta_e_ciede2000(lab_a, lab_b)
    return {
        "ssim": round(float(ssim), 4),
        "delta_e_mean": round(float(de.mean()), 3),
        "delta_e_p95": round(float(np.percentile(de, 95)), 3),
    }
