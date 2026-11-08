"""Output-size templates and geometry for the three resize modes.

A template is a width plus an aspect ratio (PRD: "宽为 ××××，宽高比 10 比 ××"), or an
explicit width x height. Height is derived as round-half-up(width * ratio_h / ratio_w).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np

from .. import config


class SizeError(ValueError):
    pass


@dataclass(frozen=True)
class TargetSize:
    width: int
    height: int
    exact_ratio: bool  # True if width:height equals the requested ratio exactly
    actual_ratio_h: float  # actual ratio expressed as 10:x for display
    suggested_widths: tuple[int, ...] = ()


def _dec(v) -> Decimal:
    try:
        d = Decimal(str(v))
    except Exception:
        raise SizeError(f"无效数值: {v}")
    if not d.is_finite() or d <= 0:
        raise SizeError(f"数值必须为正数: {v}")
    return d


def compute_target(width: int, ratio_w=None, ratio_h=None, height: int | None = None) -> TargetSize:
    """Resolve a template into exact output pixels.

    Either (width, ratio_w, ratio_h) or (width, height) must be given.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise SizeError("宽度必须是整数像素")
    if height is not None:
        h = int(height)
        exact, rw, rh = True, Decimal(width), Decimal(h)
    else:
        if ratio_w is None or ratio_h is None:
            raise SizeError("需要提供宽高比或高度")
        rw, rh = _dec(ratio_w), _dec(ratio_h)
        exact_h = Decimal(width) * rh / rw
        h = int(exact_h.quantize(Decimal(1), rounding=ROUND_HALF_UP))
        exact = exact_h == h

    for name, v in (("宽度", width), ("高度", h)):
        if v < config.MIN_SIDE or v > config.MAX_SIDE:
            raise SizeError(f"{name} {v}px 超出范围（{config.MIN_SIDE}–{config.MAX_SIDE}px）")
    if width * h > config.MAX_OUTPUT_PIXELS:
        raise SizeError(f"输出 {width}×{h} 超过 {config.MAX_OUTPUT_PIXELS // 1_000_000} MP 上限")

    suggestions: tuple[int, ...] = ()
    if not exact:
        # Nearest widths for which the ratio is exact (multiples of the reduced ratio's width term)
        frac = Fraction(str(rw)) / Fraction(str(rh))  # width / height
        step = frac.numerator  # width must be a multiple of this for an integer height
        if step <= config.MAX_SIDE:
            below = (width // step) * step
            cands = [c for c in (below, below + step) if config.MIN_SIDE <= c <= config.MAX_SIDE]
            suggestions = tuple(cands)
    return TargetSize(width=width, height=h, exact_ratio=exact,
                      actual_ratio_h=round(10 * h / width, 4), suggested_widths=suggestions)


# ---------------------------------------------------------------------------
# Geometry

@dataclass
class Plan:
    """How a source region maps onto the output canvas."""
    crop_box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in source pixels
    content_size: tuple[int, int]  # size the cropped source is resized to
    offset: tuple[int, int]  # where the content is pasted on the canvas
    canvas_size: tuple[int, int]
    magnification: float  # how much the source is enlarged (max over both axes)


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    """Median colour of a 2px border ring."""
    b = 2
    ring = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return np.median(ring, axis=0)


def subject_bbox(rgb: np.ndarray, alpha: np.ndarray | None = None, tol: float = 18.0
                 ) -> tuple[int, int, int, int] | None:
    """Bounding box of the foreground on a roughly uniform background.

    Returns None if the background is not uniform (natural photo), in which case
    callers should fall back to centred behaviour.
    """
    h, w = rgb.shape[:2]
    if alpha is not None:
        mask = alpha > 8
    else:
        bg = estimate_background(rgb)
        ring_dev = np.abs(np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3)]) - bg).max(axis=1)
        if np.percentile(ring_dev, 90) > tol:
            return None  # busy border: not a product-on-plain-background shot
        diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
        mask = diff > tol
    # ignore isolated noise pixels
    rows = np.where(mask.sum(axis=1) > max(1, w // 400))[0]
    cols = np.where(mask.sum(axis=0) > max(1, h // 400))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.97 * w * h:
        return None
    return box


def trim_box(rgb: np.ndarray, alpha: np.ndarray | None, margin_frac: float = 0.02
             ) -> tuple[int, int, int, int] | None:
    """Box that removes uniform borders, keeping a small margin."""
    box = subject_bbox(rgb, alpha)
    if box is None:
        return None
    h, w = rgb.shape[:2]
    m = int(round(max(box[2] - box[0], box[3] - box[1]) * margin_frac))
    return (max(0, box[0] - m), max(0, box[1] - m), min(w, box[2] + m), min(h, box[3] + m))


def plan_geometry(src_w: int, src_h: int, target: TargetSize, mode: str,
                  anchor_box: tuple[int, int, int, int] | None = None) -> Plan:
    tw, th = target.width, target.height
    if mode == "pad":
        s = min(tw / src_w, th / src_h)
        cw, ch = max(1, round(src_w * s)), max(1, round(src_h * s))
        cw, ch = min(cw, tw), min(ch, th)
        return Plan((0, 0, src_w, src_h), (cw, ch), ((tw - cw) // 2, (th - ch) // 2), (tw, th), s)
    if mode == "crop":
        s = max(tw / src_w, th / src_h)
        # Size of the source window that maps to the full canvas
        win_w, win_h = min(src_w, tw / s), min(src_h, th / s)
        if anchor_box is not None:
            cx, cy = (anchor_box[0] + anchor_box[2]) / 2, (anchor_box[1] + anchor_box[3]) / 2
        else:
            cx, cy = src_w / 2, src_h / 2
        x0 = float(np.clip(cx - win_w / 2, 0, src_w - win_w))
        y0 = float(np.clip(cy - win_h / 2, 0, src_h - win_h))
        box = (int(round(x0)), int(round(y0)), int(round(x0 + win_w)), int(round(y0 + win_h)))
        box = (box[0], box[1], max(box[0] + 1, min(src_w, box[2])), max(box[1] + 1, min(src_h, box[3])))
        return Plan(box, (tw, th), (0, 0), (tw, th), s)
    if mode == "stretch":
        return Plan((0, 0, src_w, src_h), (tw, th), (0, 0), (tw, th), max(tw / src_w, th / src_h))
    raise SizeError(f"未知的缩放模式: {mode}")
