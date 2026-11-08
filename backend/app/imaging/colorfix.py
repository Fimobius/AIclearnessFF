"""Wavelet colour fix (from StableSR): keep the model's high-frequency detail but take
the low-frequency colour/tone from the original. Guarantees no global colour drift.

result = high(content) + low(reference) = content - low(content) + low(reference)
where low() is 5 levels of a dilated 3x3 binomial blur (dilation 1, 2, 4, 8, 16).
"""
from __future__ import annotations

import cv2
import numpy as np

_K1 = np.array([0.25, 0.5, 0.25], np.float32)


def _dilated_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Separable [1,2,1]/4 kernel with dilation r, replicate border (CPU path)."""
    h, w = img.shape[:2]
    p = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REPLICATE)
    v = (p[:-2 * r] + 2 * p[r:h + r] + p[2 * r:]) * 0.25
    return (v[:, :-2 * r] + 2 * v[:, r:w + r] + v[:, 2 * r:]) * 0.25


def _low_np(img: np.ndarray, levels: int) -> np.ndarray:
    for i in range(levels):
        img = _dilated_blur(img, 2 ** i)
    return img


def _low_torch(img, levels: int):
    """img: (1, 3, H, W) float torch tensor on any device."""
    import torch
    k = torch.tensor(_K1, device=img.device)
    kv = k.view(1, 1, 3, 1).repeat(3, 1, 1, 1)
    kh = k.view(1, 1, 1, 3).repeat(3, 1, 1, 1)
    for i in range(levels):
        r = 2 ** i
        img = torch.nn.functional.pad(img, (r, r, r, r), mode="replicate")
        img = torch.nn.functional.conv2d(img, kv, dilation=(r, 1), groups=3)
        img = torch.nn.functional.conv2d(img, kh, dilation=(1, r), groups=3)
    return img


def wavelet_color_fix(content: np.ndarray, reference: np.ndarray, levels: int = 5,
                      device=None) -> np.ndarray:
    """content: SR output; reference: original resized to the same size. Both uint8 RGB."""
    if content.shape != reference.shape:
        raise ValueError("shape mismatch")
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch
        try:
            to_t = lambda a: torch.from_numpy(np.array(a)).to(device).permute(2, 0, 1)[None].float()  # noqa: E731
            c, r = to_t(content), to_t(reference)
            with torch.inference_mode():
                out = c - _low_torch(c, levels) + _low_torch(r, levels)
                return out.add_(0.5).clamp_(0, 255)[0].permute(1, 2, 0).byte().cpu().numpy()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()  # fall through to CPU
    c = content.astype(np.float32)
    out = c - _low_np(c, levels) + _low_np(reference.astype(np.float32), levels)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
