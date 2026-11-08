import io

import numpy as np
import pytest
from PIL import Image

from app.imaging import io as imgio
from conftest import make_image


def test_probe_accepts_supported_formats():
    for fmt in ("JPEG", "PNG", "WEBP"):
        f, w, h = imgio.probe(make_image(fmt=fmt))
        assert (f, w, h) == (fmt, 96, 64)


@pytest.mark.parametrize("data,msg", [
    (b"not an image at all", "不是有效的图片"),
    (make_image(fmt="BMP"), "不支持的格式"),
    (make_image(fmt="GIF"), "不支持的格式"),
    (make_image()[:300], "损坏"),
    (make_image(fmt="PNG")[:-40], "损坏"),
])
def test_probe_rejects_bad_files(data, msg):
    with pytest.raises(imgio.ImageError, match=msg):
        imgio.probe(data)


def test_exif_orientation_applied():
    im = Image.open(io.BytesIO(make_image(96, 64)))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif.tobytes())
    loaded = imgio.load(buf.getvalue())
    assert loaded.size == (64, 96)


def test_modes_normalised_to_rgb():
    for mode, fmt in (("L", "PNG"), ("P", "PNG"), ("CMYK", "JPEG"), ("LA", "PNG")):
        img = imgio.load(make_image(mode=mode, fmt=fmt))
        assert img.rgb.shape == (64, 96, 3) and img.rgb.dtype == np.uint8


def test_16bit_png():
    arr = (np.linspace(0, 65535, 64 * 96).reshape(64, 96)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    img = imgio.load(buf.getvalue())
    assert img.rgb.max() > 250 and any("16 位" in w for w in img.warnings)


def test_alpha_kept_only_when_used():
    rgba = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    assert imgio.load(buf.getvalue()).alpha is None
    rgba.putpixel((0, 0), (0, 0, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    assert imgio.load(buf.getvalue()).alpha is not None


COLOR_DIR = "C:/Windows/System32/spool/drivers/color"


def test_srgb_profile_passthrough():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 30)).save(buf, "PNG", icc_profile=imgio.SRGB_ICC)
    img = imgio.load(buf.getvalue())
    assert tuple(img.rgb[0, 0]) == (10, 200, 30)


def test_adobe_rgb_converted_to_srgb():
    from pathlib import Path
    prof = Path(COLOR_DIR) / "AdobeRGB1998.icc"
    if not prof.exists():
        pytest.skip("AdobeRGB profile not available")
    buf = io.BytesIO()
    # Pure Adobe-RGB green is outside sRGB; after conversion red/blue must rise from 0
    Image.new("RGB", (8, 8), (0, 255, 0)).save(buf, "JPEG", quality=100, icc_profile=prof.read_bytes())
    img = imgio.load(buf.getvalue())
    r, g, b = img.rgb[4, 4]
    assert g > 200 and any("sRGB" in w for w in img.warnings)


def test_cmyk_with_profile_converted():
    from pathlib import Path
    prof = Path(COLOR_DIR) / "CoatedFOGRA39.icc"
    if not prof.exists():
        pytest.skip("FOGRA39 profile not available")
    buf = io.BytesIO()
    Image.new("CMYK", (8, 8), (0, 0, 0, 0)).save(buf, "JPEG", quality=100, icc_profile=prof.read_bytes())
    img = imgio.load(buf.getvalue())
    assert img.rgb.shape == (8, 8, 3) and img.rgb.mean() > 230  # paper white


@pytest.mark.parametrize("fmt", ["jpg", "png", "webp"])
def test_encode_formats_open_and_carry_srgb(fmt):
    rgb = imgio.load(make_image(200, 120)).rgb
    res = imgio.encode(rgb, None, fmt, 90)
    im = Image.open(io.BytesIO(res.data))
    im.load()
    assert im.size == (200, 120)
    assert im.info.get("icc_profile") == imgio.SRGB_ICC


def test_encode_jpg_flattens_alpha():
    rgb = np.zeros((10, 10, 3), np.uint8)
    alpha = np.zeros((10, 10), np.uint8)
    res = imgio.encode(rgb, alpha, "jpg", 95, flatten_color=(255, 255, 255))
    im = Image.open(io.BytesIO(res.data)).convert("RGB")
    assert np.asarray(im).mean() > 250
    assert res.warnings


def test_encode_max_kb():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)
    res = imgio.encode(rgb, None, "jpg", 95, max_kb=150)
    assert len(res.data) <= 150 * 1024 and res.quality_used < 95


def test_parse_color():
    assert imgio.parse_color("white") == (255, 255, 255)
    assert imgio.parse_color("#FF8000") == (255, 128, 0)
    assert imgio.parse_color("transparent") is None
    with pytest.raises(ValueError):
        imgio.parse_color("#12")
