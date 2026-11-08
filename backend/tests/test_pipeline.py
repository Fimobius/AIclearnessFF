"""Pipeline tests run the real models on small images (GPU if available)."""
import io

import numpy as np
import pytest
from PIL import Image

from app.imaging import colorfix
from app.imaging.pipeline import ProcessSettings, process
from conftest import make_image


def _open(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


@pytest.mark.parametrize("tier", ["standard", "strong"])
@pytest.mark.parametrize("mode", ["pad", "crop", "stretch"])
@pytest.mark.parametrize("width,rw,rh", [(500, 10, 10), (501, 10, 3), (300, 3, 4)])
def test_output_size_is_exact(engine, tier, mode, width, rw, rh):
    s = ProcessSettings(tier=tier, width=width, ratio_w=rw, ratio_h=rh, mode=mode)
    res = process(make_image(120, 80), s, engine)
    im = _open(res.data)
    assert im.size == (s.target().width, s.target().height)


@pytest.mark.parametrize("fmt,bg,expect_mode", [("png", "transparent", "RGBA"), ("webp", "transparent", "RGBA"),
                                                ("jpg", "transparent", "RGB"), ("png", "white", "RGB")])
def test_pad_backgrounds(engine, fmt, bg, expect_mode):
    s = ProcessSettings(width=200, ratio_w=10, ratio_h=10, mode="pad", background=bg, format=fmt)
    res = process(make_image(120, 60), s, engine)
    im = _open(res.data)
    assert im.mode == expect_mode
    corner = im.convert("RGBA").getpixel((0, 0))
    if bg == "transparent" and fmt != "jpg":
        assert corner[3] == 0
    else:
        assert corner[:3] == (255, 255, 255)


def test_custom_background_colour(engine):
    s = ProcessSettings(width=200, ratio_w=10, ratio_h=10, mode="pad", background="#204080", format="png")
    im = _open(process(make_image(120, 60), s, engine).data).convert("RGB")
    assert im.getpixel((0, 0)) == (0x20, 0x40, 0x80)


def test_colour_preserved_and_report(engine):
    s = ProcessSettings(width=384, ratio_w=10, ratio_h=10, mode="stretch", format="png")
    res = process(make_image(96, 96), s, engine)
    rep = res.report
    assert rep["qa"]["delta_e_mean"] < 3.0
    assert rep["qa"]["ssim"] > 0.85
    assert rep["model"]["name"] and rep["timings"]["total"] > 0
    assert res.baseline_preview[:4] == b"RIFF"


def test_downscale_large_source(engine):
    s = ProcessSettings(width=200, ratio_w=10, ratio_h=10)
    res = process(make_image(1600, 1600), s, engine)
    assert _open(res.data).size == (200, 200)
    assert res.report["model"]["working_size"][0] < 1600  # never feeds the full image to the model


def test_large_magnification_warns(engine):
    s = ProcessSettings(width=1000, ratio_w=10, ratio_h=10)
    res = process(make_image(100, 100), s, engine)
    assert "over_model_scale" in res.report["flags"]


def test_trim_border(engine):
    arr = np.full((400, 400, 3), 255, np.uint8)
    arr[150:250, 150:250] = (0, 0, 200)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    s = ProcessSettings(width=300, ratio_w=1, ratio_h=1, trim_border=True, format="png")
    im = np.asarray(_open(process(buf.getvalue(), s, engine).data).convert("RGB"))
    blue = (im[..., 2] > 150) & (im[..., 0] < 80)
    assert blue.mean() > 0.8  # subject now fills the frame


def test_wavelet_color_fix_restores_tone():
    rng = np.random.default_rng(0)
    ref = np.full((128, 128, 3), (180, 120, 60), np.uint8)
    content = np.clip(ref.astype(int) + (40, -30, 20) + rng.normal(0, 5, ref.shape), 0, 255).astype(np.uint8)
    fixed = colorfix.wavelet_color_fix(content, ref)
    assert np.abs(fixed.mean((0, 1)) - ref.mean((0, 1))).max() < 2


def test_settings_validation():
    with pytest.raises(ValueError):
        ProcessSettings(width=100, ratio_w=10, ratio_h=2000)
    with pytest.raises(ValueError):
        ProcessSettings(background="purple")
    with pytest.raises(ValueError):
        ProcessSettings(quality=101)


def _text_image(font_px: int, w=1024, h=512, scale=4) -> bytes:
    """Label-like text rendered at HR then shrunk x`scale` (source glyph = font_px/scale)."""
    from PIL import ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    from conftest import cjk_font
    f = cjk_font(font_px)
    d.text((40, 40), "净含量 500g 6901234567892", font=f, fill=(20, 20, 20))
    d.text((40, 40 + font_px * 2), "生产日期 2026-11-08", font=f, fill=(150, 20, 20))
    im = im.resize((w // scale, h // scale), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_small_text_is_protected(engine):
    s = ProcessSettings(width=1024, ratio_w=2, ratio_h=1, mode="stretch", format="png")
    rep = process(_text_image(28), s, engine).report  # ~7 px glyphs in the source
    assert rep["text"]["protected"] >= 1
    assert "text_protected" in rep["flags"]
    assert rep["text"]["protected_boxes"] and len(rep["text"]["protected_boxes"][0]) == 4


def test_large_text_left_to_ai(engine):
    s = ProcessSettings(width=1024, ratio_w=2, ratio_h=1, mode="stretch", format="png")
    rep = process(_text_image(96), s, engine).report  # ~24 px glyphs in the source
    assert rep["text"]["regions"] >= 1 and rep["text"]["protected"] == 0
    assert "text_present" in rep["flags"]


def test_text_protect_can_be_disabled(engine):
    s = ProcessSettings(width=1024, ratio_w=2, ratio_h=1, mode="stretch", text_protect=False)
    assert process(_text_image(28), s, engine).report["text"] is None


def test_protect_blends_safe_into_text_regions():
    from app.imaging import textguard
    src = np.asarray(Image.open(io.BytesIO(_text_image(28))).convert("RGB"))
    h, w = src.shape[:2]
    sr = np.zeros((h * 4, w * 4, 3), np.uint8)  # "AI output" = black everywhere
    safe = np.full_like(sr, 255)  # "safe" = white everywhere
    out, info = textguard.protect(src, sr, safe)
    assert info["protected"] >= 1
    assert out.max() == 255 and out.mean() < 128  # white only inside protected text, rest untouched


def test_text_detector_loads_from_non_ascii_path(tmp_path, monkeypatch):
    """Users unzip the app into folders like 桌面\图片工具 - OpenCV can't open such paths."""
    import shutil
    from app import config
    from app.imaging import textguard
    d = tmp_path / "测试 目录"
    d.mkdir()
    shutil.copy2(config.MODELS_DIR / textguard.MODEL_FILE, d / textguard.MODEL_FILE)
    monkeypatch.setattr(config, "MODELS_DIR", d)
    det = textguard.TextDetector()
    src = np.asarray(Image.open(io.BytesIO(_text_image(48))).convert("RGB"))
    assert len(det.detect(src)) >= 1
