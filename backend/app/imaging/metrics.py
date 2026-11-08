"""Image-quality metrics in plain NumPy/OpenCV (no SciPy / scikit-image at runtime).

Implementations follow scikit-image so results match it (verified in tests/test_metrics.py):
- ssim: Wang et al. 2004, 7x7 uniform window, sample covariance, K1=0.01, K2=0.03
- rgb2lab: sRGB (D65) -> CIE L*a*b*
- delta_e_ciede2000: Sharma, Wu & Dalal 2005
"""
from __future__ import annotations

import cv2
import numpy as np


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0, win: int = 7) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    n = win * win
    cov_norm = n / (n - 1)

    def box(z: np.ndarray) -> np.ndarray:
        return cv2.blur(z, (win, win), borderType=cv2.BORDER_REFLECT)

    ux, uy = box(x), box(y)
    vx = cov_norm * (box(x * x) - ux * ux)
    vy = cov_norm * (box(y * y) - uy * uy)
    vxy = cov_norm * (box(x * y) - ux * uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    pad = (win - 1) // 2
    return float(s[pad:-pad, pad:-pad].mean())


_M = np.array([[0.412453, 0.357580, 0.180423],
               [0.212671, 0.715160, 0.072169],
               [0.019334, 0.119193, 0.950227]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or [0,1] float sRGB (H, W, 3) -> L*a*b* float64."""
    c = rgb.astype(np.float64)
    if rgb.dtype == np.uint8:
        c /= 255.0
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> np.ndarray:
    L1, a1, b1 = np.moveaxis(lab1.astype(np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(lab2.astype(np.float64), -1, 0)
    cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25.0 ** 7)))
    scale = 1 + G
    C1, C2 = np.hypot(a1 * scale, b1), np.hypot(a2 * scale, b2)
    h1 = np.arctan2(b1, a1 * scale) % (2 * np.pi)
    h2 = np.arctan2(b2, a2 * scale) % (2 * np.pi)

    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    Cbar = 0.5 * (C1 + C2)
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2
    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5

    T = (1 - 0.17 * np.cos(Hbar - np.deg2rad(30)) + 0.24 * np.cos(2 * Hbar)
         + 0.32 * np.cos(3 * Hbar + np.deg2rad(6)) - 0.20 * np.cos(4 * Hbar - np.deg2rad(63)))
    SH = 1 + 0.015 * Cbar * T
    H_term = dH_term / (kH * SH)

    c7 = Cbar ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25.0 ** 7))
    dtheta = np.deg2rad(30) * np.exp(-(((np.rad2deg(Hbar) - 275) / 25) ** 2))
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    return np.sqrt(L_term ** 2 + C_term ** 2 + H_term ** 2 + R_term)
