"""Visual before/after of text protection on the golden product images.

Writes report/text_protection.png: rows = image/degradation, columns = input, each tier
without and with protection, ground truth. Crops are the most text-dense 256px window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.imaging import textguard  # noqa: E402
from app.imaging.colorfix import wavelet_color_fix  # noqa: E402
from app.imaging.pipeline import TIERS  # noqa: E402
from app.imaging.sr import get_engine  # noqa: E402

ROWS = [("prod_box_text", "compressed"), ("prod_box_text", "heavy"), ("prod_bottle", "compressed"),
        ("prod_bottle", "defocus"), ("prod_electronics", "noisy")]
CROPS = {"prod_box_text": (260, 560, 660, 800), "prod_bottle": (360, 450, 680, 700), "prod_electronics": (240, 420, 800, 500)}
S = 230


def main() -> None:
    eng = get_engine()
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    cols = ["输入", *[f"{t['label']}" for t in TIERS.values()], *[f"{t['label']}+文字保护" for t in TIERS.values()], "原图"]
    out = Image.new("RGB", (S * len(cols), 24 + (S + 22) * len(ROWS)), "white")
    d = ImageDraw.Draw(out)
    for j, c in enumerate(cols):
        d.text((j * S + 6, 3), c, font=font, fill="black")
    for i, (name, deg) in enumerate(ROWS):
        pair = ROOT / "eval" / "golden" / "pairs" / name
        hr = np.asarray(Image.open(pair / "hr.png").convert("RGB"))
        lr = np.asarray(Image.open(pair / f"{deg}.png").convert("RGB"))
        H, W = hr.shape[:2]
        lanczos = np.asarray(Image.fromarray(lr).resize((W, H), Image.Resampling.LANCZOS))
        safe = textguard.conservative(lr, (W, H))
        plain, prot = [], []
        for t in TIERS.values():
            sr = wavelet_color_fix(eng.upscale(lr, t["spec"]), lanczos, device=eng.torch_device)
            plain.append(sr)
            prot.append(textguard.protect(lr, sr, safe, t["text_px"])[0])
        nearest = np.asarray(Image.fromarray(lr).resize((W, H), Image.Resampling.NEAREST))
        imgs = [nearest, *plain, *prot, hr]
        box = CROPS[name]
        for j, im in enumerate(imgs):
            tile = Image.fromarray(im).crop(box)
            tile.thumbnail((S - 4, S - 4), Image.Resampling.LANCZOS)
            out.paste(tile, (j * S + 2, 24 + i * (S + 22)))
        d.text((6, 24 + i * (S + 22) + S + 2), f"{name} / {deg}", font=font, fill="black")
    dest = ROOT / "eval" / "report" / "text_protection.png"
    out.save(dest)
    print(dest)


if __name__ == "__main__":
    main()
