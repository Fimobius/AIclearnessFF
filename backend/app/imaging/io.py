"""Decoding, validation, colour normalisation and encoding of images.

Everything inside the pipeline works on sRGB uint8 numpy arrays (H, W, 3) plus an
optional alpha plane (H, W). This module is the only place that deals with file
formats, ICC profiles, EXIF orientation and odd pixel modes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .. import config

Image.MAX_IMAGE_PIXELS = config.MAX_INPUT_PIXELS

ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(_SRGB).tobytes()


class ImageError(Exception):
    """User-facing validation error (message is shown in the UI)."""


@dataclass
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8, sRGB
    alpha: np.ndarray | None  # (H, W) uint8 or None
    format: str  # JPEG / PNG / WEBP
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


def probe(data: bytes) -> tuple[str, int, int]:
    """Cheap validation used at upload time. Returns (format, width, height)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in ALLOWED_FORMATS:
                raise ImageError(f"不支持的格式 {fmt}，仅支持 JPG / PNG / WEBP")
            im.verify()
        # verify() leaves the file unusable; reopen and fully decode to catch truncation
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.size
    except ImageError:
        raise
    except Image.DecompressionBombError:
        raise ImageError(f"图片像素过大（上限 {config.MAX_INPUT_PIXELS // 1_000_000} MP）")
    except UnidentifiedImageError:
        raise ImageError("不是有效的图片文件")
    except Exception as e:  # truncated / corrupt data raises OSError, SyntaxError, ...
        raise ImageError(f"图片已损坏，无法解码（{type(e).__name__}）")
    return fmt, w, h


def _estimate_jpeg_quality(im: Image.Image) -> int | None:
    """Estimate the libjpeg quality factor from the luminance quantisation table."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    # Mean ratio is independent of zigzag vs natural ordering of the table
    scale = float(np.mean(np.asarray(q[0], float)) / std.mean()) * 100  # inverse of libjpeg scaling
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(np.clip(round(quality), 1, 100))


def load(data: bytes) -> LoadedImage:
    probe(data)
    warnings: list[str] = []
    im = Image.open(io.BytesIO(data))
    fmt = im.format
    info: dict = {"format": fmt, "mode": im.mode, "orig_size": list(im.size)}
    if fmt == "JPEG":
        info["jpeg_quality"] = _estimate_jpeg_quality(im)

    if getattr(im, "n_frames", 1) > 1:
        warnings.append("动图仅处理第一帧")
        im.seek(0)

    icc = im.info.get("icc_profile")
    orientation = im.getexif().get(0x0112, 1)
    if orientation != 1:
        im = ImageOps.exif_transpose(im)
        info["exif_orientation"] = orientation

    # Normalise exotic modes
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0) * 255.0
        im = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "L")
        warnings.append("16 位图已转换为 8 位")
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    if im.mode == "LA" or im.mode == "PA":
        im = im.convert("RGBA")
    if im.mode in ("1", "L", "F"):
        im = im.convert("RGB")

    alpha = None
    if im.mode == "RGBA":
        alpha_img = im.getchannel("A")
        a = np.asarray(alpha_img)
        if a.min() < 255:
            alpha = a.copy()
        im = im.convert("RGB")

    # Colour management: convert any embedded profile (including CMYK) to sRGB
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            desc = ImageCms.getProfileDescription(src).strip()
            info["icc"] = desc
            if "srgb" not in desc.lower():
                im = ImageCms.profileToProfile(im, src, _SRGB, outputMode="RGB",
                                               renderingIntent=ImageCms.Intent.PERCEPTUAL)
                warnings.append(f"已将色彩配置 {desc} 转换为 sRGB")
        except Exception:
            warnings.append("嵌入的色彩配置无法解析，按 sRGB 处理")
    if im.mode != "RGB":  # CMYK without profile, YCbCr, ...
        im = im.convert("RGB")

    rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    return LoadedImage(rgb=rgb, alpha=alpha, format=fmt, warnings=warnings, info=info)


def make_thumbnail(data: bytes, max_side: int = 360) -> bytes:
    img = load(data)
    im = to_pil(img.rgb, img.alpha)
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=80)
    return buf.getvalue()


def to_pil(rgb: np.ndarray, alpha: np.ndarray | None = None) -> Image.Image:
    im = Image.fromarray(rgb, "RGB")
    if alpha is not None:
        im.putalpha(Image.fromarray(alpha, "L"))
    return im


def parse_color(value: str) -> tuple[int, int, int] | None:
    """'white' | 'black' | 'transparent' | '#rrggbb' -> RGB tuple, None for transparent."""
    v = (value or "white").strip().lower()
    named = {"white": (255, 255, 255), "black": (0, 0, 0)}
    if v == "transparent":
        return None
    if v in named:
        return named[v]
    if v.startswith("#") and len(v) == 7:
        try:
            return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
        except ValueError:
            pass
    raise ValueError(f"无效的颜色值: {value}")


@dataclass
class EncodeResult:
    data: bytes
    ext: str
    quality_used: int | None
    warnings: list[str]


def _encode_once(im: Image.Image, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpg":
        kw = dict(quality=quality, subsampling=0 if quality >= 90 else 2, icc_profile=SRGB_ICC)
        try:
            im.save(buf, "JPEG", optimize=True, progressive=True, **kw)
        except OSError:
            # Pillow sizes the progressive buffer at ~1 byte/pixel; very noisy 4:4:4 images
            # can exceed it ("Suspension not allowed"). Baseline encoding has no such limit.
            buf = io.BytesIO()
            im.save(buf, "JPEG", **kw)
    elif fmt == "webp":
        im.save(buf, "WEBP", quality=quality, method=4, icc_profile=SRGB_ICC)
    elif fmt == "png":
        im.save(buf, "PNG", compress_level=6, icc_profile=SRGB_ICC)
    else:
        raise ValueError(fmt)
    return buf.getvalue()


def encode(rgb: np.ndarray, alpha: np.ndarray | None, fmt: str, quality: int = 92,
           max_kb: int | None = None, flatten_color: tuple[int, int, int] = (255, 255, 255)) -> EncodeResult:
    warnings: list[str] = []
    im = to_pil(rgb, alpha)
    if fmt == "jpg" and alpha is not None:
        bg = Image.new("RGB", im.size, flatten_color)
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
        warnings.append("JPG 不支持透明，透明区域已填充底色")

    quality = int(np.clip(quality, 0, 100))
    data = _encode_once(im, fmt, quality)
    used = quality if fmt != "png" else None
    if max_kb and len(data) > max_kb * 1024:
        if fmt == "png":
            warnings.append(f"PNG 为无损格式，无法压缩到 {max_kb} KB 以内（实际 {len(data) // 1024} KB）")
        else:
            # Binary search the highest quality that fits
            lo, hi, best = 10, quality - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                d = _encode_once(im, fmt, mid)
                if len(d) <= max_kb * 1024:
                    best, lo = (mid, d), mid + 1
                else:
                    hi = mid - 1
            if best:
                used, data = best
                if used < quality:
                    warnings.append(f"为满足 {max_kb} KB 上限，画质由 {quality} 降至 {used}")
            else:
                data = _encode_once(im, fmt, 10)
                used = 10
                warnings.append(f"画质降到 10 仍超过 {max_kb} KB（实际 {len(data) // 1024} KB）")
    return EncodeResult(data=data, ext=fmt, quality_used=used, warnings=warnings)
