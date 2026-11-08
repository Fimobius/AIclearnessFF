"""End-to-end processing of one image: decode -> crop/plan -> SR -> colour lock -> resize
-> fidelity check -> compose canvas -> encode."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .. import config
from . import analyze, colorfix, io, sizing, textguard
from .sr import ModelSpec, SREngine

# Enhancement tiers, chosen from the model bake-off (eval/README.md):
#  standard: best fidelity (PSNR/SSIM/ΔE) of all candidates, least invented detail
#  strong:   best perceptual quality (LPIPS); beats Real-ESRGAN x4plus on every metric
# text_px: small-text protection threshold (detector box height in model-input px),
#  calibrated by OCR in eval/text_eval.py - the GAN invents glyphs at larger sizes.
TIERS: dict[str, dict] = {
    "standard": {"label": "标准增强", "spec": ModelSpec("SwinIR-M_x4_PSNR"), "text_px": 11.0,
                 "desc": "保留原图质感，轻度降噪 + 高清化（适合产品实拍图）"},
    "strong": {"label": "强力修复", "spec": ModelSpec("SwinIR-M_x4_GAN"), "text_px": 13.0,
               "desc": "重度模糊、老旧图片，更多细节还原"},
}

# SR output is kept at most this large before the final resize (memory bound)
MAX_SR_OUTPUT_PIXELS = 48_000_000


class ProcessSettings(BaseModel):
    tier: Literal["standard", "strong"] = "standard"
    width: int = Field(1200, ge=config.MIN_SIDE, le=config.MAX_SIDE)
    ratio_w: float | None = Field(10, gt=0)
    ratio_h: float | None = Field(10, gt=0)
    height: int | None = Field(None, ge=config.MIN_SIDE, le=config.MAX_SIDE)
    mode: Literal["pad", "crop", "stretch"] = "pad"
    background: str = "white"
    crop_anchor: Literal["center", "subject"] = "subject"
    trim_border: bool = False
    text_protect: bool = True
    format: Literal["jpg", "png", "webp"] = "jpg"
    quality: int = Field(92, ge=0, le=100)
    max_kb: int | None = Field(None, ge=10)
    name_suffix: str = Field("_hd", max_length=40)

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "stretch" and not config.ALLOW_STRETCH:
            raise ValueError("拉伸模式已被管理员关闭")
        io.parse_color(self.background)
        self.target()
        return self

    def target(self) -> sizing.TargetSize:
        if self.height is not None:
            return sizing.compute_target(self.width, height=self.height)
        return sizing.compute_target(self.width, self.ratio_w, self.ratio_h)


@dataclass
class ProcessResult:
    data: bytes
    ext: str
    width: int
    height: int
    baseline_preview: bytes = b""  # webp of the same framing using plain Lanczos
    report: dict = field(default_factory=dict)


ProgressFn = Callable[[str, float], None]


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))


def _resize_plane(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (a.shape[1], a.shape[0]) == size:
        return a
    return np.asarray(Image.fromarray(a, "L").resize(size, Image.Resampling.LANCZOS))


def _compose(content: np.ndarray, alpha: np.ndarray | None, mode: str, plan: sizing.Plan,
             size: tuple[int, int], bg: tuple[int, int, int] | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Place the resized content on the output canvas. Returns (rgb, alpha-or-None)."""
    tw, th = size
    ch, cw = content.shape[:2]
    ox, oy = plan.offset

    def flatten(rgb: np.ndarray, a: np.ndarray, color) -> np.ndarray:
        f = a[..., None].astype(np.float32) / 255
        return (rgb * f + np.array(color, np.float32) * (1 - f) + 0.5).astype(np.uint8)

    if mode != "pad":
        if alpha is not None and bg is not None:
            return flatten(content, alpha, bg), None
        return content, alpha
    if bg is None:  # transparent padding
        canvas = np.zeros((th, tw, 3), np.uint8)
        out_alpha = np.zeros((th, tw), np.uint8)
        canvas[oy:oy + ch, ox:ox + cw] = content
        out_alpha[oy:oy + ch, ox:ox + cw] = alpha if alpha is not None else 255
        return canvas, out_alpha
    canvas = np.empty((th, tw, 3), np.uint8)
    canvas[:] = bg
    canvas[oy:oy + ch, ox:ox + cw] = flatten(content, alpha, bg) if alpha is not None else content
    return canvas, None


def sr_workload(src_w: int, src_h: int, settings: ProcessSettings, engine: SREngine) -> float:
    """Megapixels the SR model will produce for a source of this size (for time estimates)."""
    plan = sizing.plan_geometry(src_w, src_h, settings.target(), settings.mode)
    x0, y0, x1, y1 = plan.crop_box
    sw, sh = x1 - x0, y1 - y0
    cw, ch = plan.content_size
    scale = engine.scale_of(TIERS[settings.tier]["spec"])
    work = min(1.0, engine.supersample * max(cw / sw, ch / sh) / scale)
    out_px = sw * sh * (work * scale) ** 2
    return min(out_px, MAX_SR_OUTPUT_PIXELS) / 1e6


def process(data: bytes, settings: ProcessSettings, engine: SREngine,
            progress: ProgressFn | None = None) -> ProcessResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    prog = progress or (lambda stage, frac: None)

    def lap(name: str, start: float) -> float:
        now = time.perf_counter()
        timings[name] = round(now - start, 3)
        return now

    prog("解码", 0.02)
    img = io.load(data)
    warnings = list(img.warnings)
    flags: list[str] = []
    diag = analyze.diagnose(img.rgb)
    target = settings.target()
    t = lap("decode", t0)

    rgb, alpha = img.rgb, img.alpha
    trim = None
    if settings.trim_border:
        box = trim = sizing.trim_box(rgb, alpha)
        if box:
            rgb = rgb[box[1]:box[3], box[0]:box[2]]
            alpha = alpha[box[1]:box[3], box[0]:box[2]] if alpha is not None else None
        else:
            warnings.append("未检测到可裁掉的纯色边框")

    sh, sw = rgb.shape[:2]
    anchor = None
    if settings.mode == "crop" and settings.crop_anchor == "subject":
        anchor = sizing.subject_bbox(rgb, alpha)
        if anchor is None:
            warnings.append("未识别到纯色背景上的主体，改为居中裁剪")
    plan = sizing.plan_geometry(sw, sh, target, settings.mode, anchor)
    x0, y0, x1, y1 = plan.crop_box
    src = rgb[y0:y1, x0:x1]
    src_alpha = alpha[y0:y1, x0:x1] if alpha is not None else None
    sh, sw = src.shape[:2]
    cw, ch = plan.content_size
    mag = max(cw / sw, ch / sh)

    # --- super-resolution -------------------------------------------------
    tier = TIERS[settings.tier]
    spec: ModelSpec = tier["spec"]
    scale = engine.scale_of(spec)
    # Feed the model an input such that its output is ~2x the needed content size
    # (supersampling suppresses GAN artefacts; 1x on CPU-only machines); never enlarge the input.
    work = min(1.0, engine.supersample * mag / scale)
    if sw * sh * (work * scale) ** 2 > MAX_SR_OUTPUT_PIXELS:
        work = min(work, (MAX_SR_OUTPUT_PIXELS / (sw * sh)) ** 0.5 / scale)
    ww, wh = max(1, round(sw * work)), max(1, round(sh * work))
    working = _resize(src, (ww, wh))
    if mag > scale:
        flags.append("over_model_scale")
        warnings.append(f"需要放大 {mag:.1f} 倍，超过模型的 {scale} 倍，超出部分为插值放大，细节有限")
    t = lap("prepare", t)

    prog("AI 高清化", 0.05)
    sr_out = engine.upscale(working, spec, lambda d, n: prog("AI 高清化", 0.05 + 0.8 * d / n))
    t = lap("super_resolution", t)

    prog("色彩保护", 0.86)
    ref = _resize(src, (sr_out.shape[1], sr_out.shape[0]))
    sr_out = colorfix.wavelet_color_fix(sr_out, ref, device=engine.torch_device)
    content = _resize(sr_out, (cw, ch))
    content_alpha = _resize_plane(src_alpha, (cw, ch)) if src_alpha is not None else None
    t = lap("color_resize", t)

    text_info = None
    if settings.text_protect and textguard.available():
        prog("文字保护", 0.88)
        # Text size is judged in the image the model actually saw (`working`); the safe
        # replacement is rendered from the full-resolution source.
        safe = textguard.conservative(src, (cw, ch))
        content, text_info = textguard.protect(working, content, safe, tier["text_px"])
        if text_info["protected"]:
            flags.append("text_protected")
            warnings.append(f"检测到 {text_info['protected']} 处文字在原图中过小或过于模糊、AI 无法可靠还原，"
                            "已保留原图笔画（未让 AI 补画）")
        elif text_info["regions"]:
            flags.append("text_present")
        ox, oy = plan.offset if settings.mode == "pad" else (0, 0)
        text_info["protected_boxes"] = [[[round(x + ox, 1), round(y + oy, 1)] for x, y in b["quad"]]
                                        for b in text_info.pop("boxes") if b["protected"]][:100]
        t = lap("text_protect", t)

    prog("质量校验", 0.9)
    fid = analyze.fidelity(src, content)
    baseline = _resize(src, (cw, ch))
    small = lambda a: analyze._limit(a, 1024)  # noqa: E731
    sharp_gain = analyze.sharpness(small(content)) / max(analyze.sharpness(small(baseline)), 1e-6)
    needs_review = fid["delta_e_mean"] > config.QA_MAX_DELTA_E or fid["ssim"] < config.QA_MIN_SSIM
    if needs_review:
        flags.append("needs_review")
    t = lap("qa", t)

    # --- compose canvas ---------------------------------------------------
    prog("输出", 0.95)
    tw, th = target.width, target.height
    bg = io.parse_color(settings.background)
    canvas, out_alpha = _compose(content, content_alpha, settings.mode, plan, (tw, th), bg)
    if canvas.shape[:2] != (th, tw):  # hard guarantee from the acceptance criteria
        raise RuntimeError(f"输出尺寸 {canvas.shape[1]}×{canvas.shape[0]} 与目标 {tw}×{th} 不一致")
    # Same framing without AI, for the before/after comparison in the UI
    base_canvas, base_alpha = _compose(baseline, content_alpha, settings.mode, plan, (tw, th), bg)
    baseline_preview = io.encode(base_canvas, base_alpha, "webp", 90).data

    enc = io.encode(canvas, out_alpha, settings.format, settings.quality, settings.max_kb,
                    flatten_color=bg or (255, 255, 255))
    warnings += enc.warnings
    if settings.format == "jpg" and (out_alpha is not None):
        flags.append("transparent_as_jpg")
    lap("encode", t)
    timings["total"] = round(time.perf_counter() - t0, 3)

    report = {
        "input": {**img.info, **diag},
        "target": {"width": tw, "height": th, "exact_ratio": target.exact_ratio,
                   "ratio": f"10:{target.actual_ratio_h:g}"},
        "geometry": {"mode": settings.mode, "trim_box": list(trim) if trim else None, "crop_box": list(plan.crop_box),
                     "content_size": [cw, ch], "offset": list(plan.offset),
                     "magnification": round(mag, 3)},
        "model": {"tier": settings.tier, "name": spec.name, "denoise": spec.denoise,
                  "scale": scale, "working_size": [ww, wh], "sr_output": [sr_out.shape[1], sr_out.shape[0]]},
        "text": text_info,
        "qa": {**fid, "sharpness_gain": round(sharp_gain, 2), "needs_review": needs_review,
               "thresholds": {"delta_e_mean_max": config.QA_MAX_DELTA_E, "ssim_min": config.QA_MIN_SSIM}},
        "output": {"format": enc.ext, "quality": enc.quality_used, "bytes": len(enc.data),
                   "has_alpha": out_alpha is not None and settings.format != "jpg"},
        "timings": timings,
        "flags": flags,
        "warnings": warnings,
    }
    prog("完成", 1.0)
    return ProcessResult(data=enc.data, ext=enc.ext, width=tw, height=th,
                         baseline_preview=baseline_preview, report=report)
