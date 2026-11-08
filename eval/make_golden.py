"""Build the golden test set: high-quality references + deterministic degradations.

HR sources:
  golden/hr_div2k/*.png   DIV2K natural photos (download separately, see README)
  golden/hr_product/*.png synthetic product shots generated here (text, logos, saturated colours,
                          fine textures on white) - the cases e-commerce images care about.
Output:
  golden/pairs/<name>/hr.png and <degradation>.png (x4 smaller)
"""
from __future__ import annotations

import io
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent / "golden"
HR = 1024  # HR crop size; LR is HR / 4
FONTS = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for f in (name, "msyh.ttc", "arial.ttf"):
        p = FONTS / f
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def _canvas(color=(255, 255, 255)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (HR, HR), color)
    return im, ImageDraw.Draw(im)


def product_box() -> Image.Image:
    im, d = _canvas()
    d.rounded_rectangle((212, 140, 812, 900), 24, fill=(232, 62, 48), outline=(150, 30, 20), width=6)
    d.rectangle((262, 250, 762, 520), fill=(255, 250, 240))
    d.text((290, 270), "有机燕麦片", font=font("msyhbd.ttc", 72), fill=(40, 40, 40))
    d.text((292, 370), "ORGANIC ROLLED OATS", font=font("arialbd.ttf", 40), fill=(60, 60, 60))
    d.text((292, 430), "净含量 Net Wt. 500g  保质期 12个月", font=font("msyh.ttc", 26), fill=(80, 80, 80))
    for i, x in enumerate(range(300, 560, 6)):  # barcode
        if (i * 7) % 3:
            d.rectangle((x, 600, x + (2 if i % 2 else 3), 760), fill=(0, 0, 0))
    d.text((300, 770), "6 901234 567892", font=font("arial.ttf", 28), fill=(0, 0, 0))
    d.ellipse((600, 600, 760, 760), fill=(30, 120, 60))
    d.text((628, 655), "ECO", font=font("arialbd.ttf", 44), fill=(255, 255, 255))
    d.text((240, 820), "配料：100% 全谷物燕麦  生产许可证号 SC10123456789012", font=font("msyh.ttc", 20), fill=(255, 255, 255))
    return im


def product_bottle() -> Image.Image:
    im, d = _canvas()
    arr = np.array(im)
    # cylindrical shading on a blue bottle
    x = np.linspace(-1, 1, 360)
    shade = (0.55 + 0.45 * np.sqrt(1 - x ** 2))[None, :, None]
    body = (np.array([40, 90, 200])[None, None, :] * shade).astype(np.uint8)
    arr[260:900, 332:692] = np.repeat(body, 640, axis=0)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((440, 120, 584, 262), 18, fill=(220, 220, 225), outline=(120, 120, 130), width=3)
    d.rectangle((352, 450, 672, 700), fill=(250, 250, 250))
    d.text((380, 470), "玻尿酸精华液", font=font("msyhbd.ttc", 44), fill=(20, 40, 110))
    d.text((382, 540), "HYALURONIC SERUM", font=font("arialbd.ttf", 30), fill=(20, 40, 110))
    d.text((382, 590), "30 mL / 1.0 fl.oz", font=font("arial.ttf", 24), fill=(60, 60, 60))
    d.text((382, 630), "深层补水 · 舒缓修护", font=font("msyh.ttc", 24), fill=(60, 60, 60))
    return im


def product_textile() -> Image.Image:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:HR, 0:HR]
    weave = ((np.sin(xx * 0.9) * np.sin(yy * 0.9)) > 0).astype(np.float32)
    check = (((xx // 64) + (yy // 64)) % 2).astype(np.float32)
    base = np.stack([150 + 60 * check, 40 + 30 * check, 60 + 20 * weave], -1)
    base += rng.normal(0, 6, base.shape)
    base += 25 * weave[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def product_electronics() -> Image.Image:
    im, d = _canvas((245, 245, 245))
    d.rounded_rectangle((180, 200, 844, 820), 40, fill=(30, 30, 34))
    for gx in range(220, 800, 12):  # speaker grille
        for gy in range(560, 780, 12):
            d.ellipse((gx, gy, gx + 5, gy + 5), fill=(90, 90, 95))
    d.rounded_rectangle((220, 240, 804, 520), 16, fill=(10, 60, 120))
    d.text((250, 270), "12:45", font=font("arialbd.ttf", 140), fill=(230, 240, 255))
    d.text((256, 440), "2026年11月8日 星期日  多云 18°C", font=font("msyh.ttc", 30), fill=(200, 210, 230))
    return im


def product_colors() -> Image.Image:
    im, d = _canvas()
    colors = [(220, 20, 60), (255, 140, 0), (255, 215, 0), (50, 205, 50), (0, 128, 128),
              (30, 144, 255), (75, 0, 130), (238, 130, 238), (139, 69, 19), (128, 128, 128),
              (255, 192, 203), (0, 0, 0)]
    for i, c in enumerate(colors):
        x, y = 112 + (i % 4) * 200, 160 + (i // 4) * 240
        d.ellipse((x, y, x + 180, y + 180), fill=c)
        d.text((x + 20, y + 190), "#%02X%02X%02X" % c, font=font("arial.ttf", 26), fill=(30, 30, 30))
    return im


PRODUCTS = {
    "prod_box_text": product_box, "prod_bottle": product_bottle, "prod_textile": product_textile,
    "prod_electronics": product_electronics, "prod_colors": product_colors,
}


# ---------------------------------------------------------------------------
# Degradations (applied to HR, produce LR = HR/4)

def _jpeg(a: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB"))


def _down(a: np.ndarray, kind=cv2.INTER_CUBIC) -> np.ndarray:
    return cv2.resize(a, (a.shape[1] // 4, a.shape[0] // 4), interpolation=kind)


def deg_compressed(a, rng):  # small, re-compressed web image
    return _jpeg(_down(a, cv2.INTER_AREA), 70)


def deg_defocus(a, rng):  # out-of-focus shot
    b = cv2.GaussianBlur(a, (0, 0), 6.0)
    return _jpeg(_down(b, cv2.INTER_AREA), 85)


def deg_noisy(a, rng):  # low light sensor noise
    lr = _down(a, cv2.INTER_AREA).astype(np.float32)
    lr += rng.normal(0, 10, lr.shape)
    return _jpeg(np.clip(lr, 0, 255).astype(np.uint8), 88)


def deg_heavy(a, rng):  # old / heavily degraded
    b = cv2.GaussianBlur(a, (0, 0), 5.0)
    lr = _down(b).astype(np.float32) + rng.normal(0, 6, (a.shape[0] // 4, a.shape[1] // 4, 3))
    return _jpeg(np.clip(lr, 0, 255).astype(np.uint8), 40)


DEGRADATIONS = {"compressed": deg_compressed, "defocus": deg_defocus, "noisy": deg_noisy, "heavy": deg_heavy}


def center_crop(im: Image.Image, n: int) -> Image.Image:
    w, h = im.size
    s = min(w, h, n)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((n, n), Image.Resampling.LANCZOS) if s != n else im


def main() -> None:
    (ROOT / "hr_product").mkdir(parents=True, exist_ok=True)
    for name, fn in PRODUCTS.items():
        fn().save(ROOT / "hr_product" / f"{name}.png")
    sources = sorted((ROOT / "hr_product").glob("*.png")) + sorted((ROOT / "hr_div2k").glob("*.png"))
    for src in sources:
        name = src.stem if src.parent.name == "hr_product" else f"div2k_{src.stem}"
        out = ROOT / "pairs" / name
        out.mkdir(parents=True, exist_ok=True)
        hr = np.asarray(center_crop(Image.open(src).convert("RGB"), HR))
        Image.fromarray(hr).save(out / "hr.png")
        for i, (dname, dfn) in enumerate(DEGRADATIONS.items()):
            rng = np.random.default_rng(zlib.crc32(name.encode()) + i)
            Image.fromarray(dfn(hr, rng)).save(out / f"{dname}.png")
    print(f"{len(sources)} HR images x {len(DEGRADATIONS)} degradations -> {ROOT / 'pairs'}", file=sys.stderr)


if __name__ == "__main__":
    main()
