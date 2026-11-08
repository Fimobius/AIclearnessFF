"""Small-text protection.

Super-resolution models cannot recover text that is too small/blurry to read in the
source; GAN models then draw stroke-like shapes that read as *different* characters
(e.g. barcode digits 6 -> 3). That is content tampering, so wherever text is too small
we keep a conservative result (plain Lanczos resize: cannot invent strokes) instead of
the model output.

Text regions = PP-OCRv3 detections on the source  ∪  detections on the SR output
(the latter catches text the model *invented* out of blobs the detector can't see in
the blurry source). A region is protected when, in the model's input,
  - its detector-box height is below the tier's threshold (too small), or
  - its edge width / box height exceeds BLUR_RATIO (big enough, but too blurred to read).
Both thresholds are calibrated with eval/text_eval.py, see eval/README.md.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .. import config

MODEL_FILE = "text_detection_cn_ppocrv3_2023may.onnx"
# Detector box height (source px) below which text is protected; per-tier values live in
# pipeline.TIERS. Box height ~ 1.4 x glyph height (11 px box ~ 7.5 px glyph).
DEFAULT_THRESHOLD_PX = 11.0
# Edge width (contrast / p99 gradient, ~2 px for a sharp edge) relative to box height.
# Text the models still read correctly measured <= 0.18; illegible blurred titles 0.31+.
BLUR_RATIO = 0.25
DET_SIDE = 960  # detector input long side (multiple of 32)


def quad_height(quad: np.ndarray) -> float:
    """Short side of the quad's minimum-area rectangle (= text line height)."""
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(quad, np.float32))
    return float(min(w, h))


class TextDetector:
    def __init__(self):
        path = config.MODELS_DIR / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(path)
        # Load from memory: OpenCV cannot open files under non-ASCII paths on Windows
        # (e.g. a Chinese user name or folder), which is where users unzip the app.
        net = cv2.dnn.readNetFromONNX(np.fromfile(path, dtype=np.uint8))
        self._model = cv2.dnn_TextDetectionModel_DB(net)
        self._model.setBinaryThreshold(0.3)
        self._model.setPolygonThreshold(0.5)
        self._model.setMaxCandidates(300)
        self._model.setUnclipRatio(2.0)
        self._lock = threading.Lock()

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Returns quads (4x2 float32) in the coordinates of `rgb`."""
        h, w = rgb.shape[:2]
        s = DET_SIDE / max(h, w)
        dw, dh = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        img = cv2.resize(rgb, (dw, dh), interpolation=interp)
        with self._lock:
            self._model.setInputParams(1.0 / 255.0, (dw, dh), (122.67891434, 116.66876762, 104.00698793), True)
            boxes, _ = self._model.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sx, sy = w / dw, h / dh
        return [np.asarray(b, np.float32) * [sx, sy] for b in boxes]


_detector: TextDetector | None = None
_det_lock = threading.Lock()


def get_detector() -> TextDetector:
    global _detector
    with _det_lock:
        if _detector is None:
            _detector = TextDetector()
        return _detector


def available() -> bool:
    return (config.MODELS_DIR / MODEL_FILE).exists()


def edge_width(gray: np.ndarray, quad: np.ndarray) -> float | None:
    """Typical edge transition width (px) of the content inside `quad`."""
    m = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(quad).round().astype(np.int32)], 1)
    inside = m > 0
    if inside.sum() < 20:
        return None
    v = gray[inside]
    contrast = float(np.percentile(v, 98) - np.percentile(v, 2))
    if contrast < 20:  # flat region, nothing to judge
        return None
    gy, gx = np.gradient(gray)
    return contrast / max(float(np.percentile(np.hypot(gx, gy)[inside], 99)), 1e-3)


@dataclass
class Region:
    quad: list[list[float]]  # in output-content coordinates
    height_src_px: float
    blur_ratio: float | None  # edge width / height, in source px
    source: str  # "source" | "ai_output"
    protected: bool
    reason: str = ""


def conservative(src: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Plain Lanczos resize. OCR on the calibration set read it slightly better than a
    sharpened version (eval/report/text_eval.md), and it cannot add strokes."""
    from PIL import Image
    return np.asarray(Image.fromarray(src).resize(size, Image.Resampling.LANCZOS))


def protect(src: np.ndarray, sr: np.ndarray, safe: np.ndarray,
            threshold_px: float = DEFAULT_THRESHOLD_PX) -> tuple[np.ndarray, dict]:
    """Blend `safe` into `sr` over small-text regions.

    src: source crop (what the model saw), sr/safe: same size, output-content resolution.
    Returns (result, info) where info lists regions and how many were protected.
    """
    det = get_detector()
    oh, ow = sr.shape[:2]
    sx, sy = ow / src.shape[1], oh / src.shape[0]  # output px per source px (differ in stretch mode)
    scale = (sx + sy) / 2
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
    regions: list[Region] = []
    found = [(q, "source") for q in det.detect(src)] + [(q / [sx, sy], "ai_output") for q in det.detect(sr)]
    for quad_src, origin in found:
        h = quad_height(quad_src)
        ew = edge_width(gray, quad_src)
        ratio = round(ew / h, 3) if ew and h > 0 else None
        regions.append(Region((quad_src * [sx, sy]).round(1).tolist(), round(h, 1), ratio, origin, False))

    mask = np.zeros((oh, ow), np.uint8)
    for r in regions:
        if r.height_src_px < threshold_px:
            r.protected, r.reason = True, "too_small"
        elif r.blur_ratio is not None and r.blur_ratio > BLUR_RATIO:
            r.protected, r.reason = True, "too_blurry"
        if r.protected:
            cv2.fillPoly(mask, [np.asarray(r.quad, np.float32).round().astype(np.int32)], 255)
    n_prot = sum(r.protected for r in regions)
    info = {"regions": len(regions), "protected": n_prot, "threshold_src_px": threshold_px, "blur_ratio": BLUR_RATIO,
            "protected_area": round(float((mask > 0).mean()), 4),
            "boxes": [r.__dict__ for r in regions][:200]}
    if n_prot == 0:
        return sr, info
    # grow slightly and feather so there is no visible seam
    k = max(3, int(round(scale * 2)) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), max(1.0, scale))[..., None]
    out = sr.astype(np.float32) * (1 - alpha) + safe.astype(np.float32) * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info
