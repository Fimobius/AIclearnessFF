"""Text-fidelity evaluation: at which source text size does AI super-resolution start
writing *wrong* characters, and does text protection prevent it?

For each glyph size, renders short strings (6-digit numbers, Chinese words, English words,
quantities like "250mL") on three 1024x1024 HR canvases with different backgrounds,
degrades them x4 (compressed / defocus / noisy), restores with each method, and reads
every string with an independent OCR (OpenCV-zoo CRNN, Chinese charset). Ground truth is
the rendered string; strings the OCR cannot read on the clean HR image are excluded.

Per (degradation, source glyph px, method):
  read_ok - OCR reads exactly the true string
  wrong   - OCR reads a *different* string of plausible length (text that says something else)
Also sweeps the protection threshold and records detector box heights per glyph size.

Usage: backend/.venv/Scripts/python eval/text_eval.py
Writes report/text_eval.json and report/text_eval.md
"""
from __future__ import annotations

import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.imaging import textguard  # noqa: E402
from app.imaging.colorfix import wavelet_color_fix  # noqa: E402
from app.imaging.pipeline import TIERS  # noqa: E402
from app.imaging.sr import get_engine  # noqa: E402

OCR_DIR = ROOT / "eval" / "ocr"
HR = 1024
SIZES_HR = [16, 24, 32, 40, 48, 56, 64, 80]  # font px at HR; source glyph = /4
FONT = "C:/Windows/Fonts/msyh.ttc"
KINDS = ("compressed", "defocus", "noisy")
BGS = [(255, 255, 255), (245, 236, 220), (225, 238, 250)]
THRESHOLDS = [11, 12, 13, 14, 16]
WORDS_CN = ["燕麦片", "精华液", "保质期", "净含量", "有机", "配料表", "生产日期", "原产地", "营养成分", "储存条件",
            "蛋白质", "碳水化合物", "脂肪", "能量", "规格", "型号", "颜色", "尺码", "材质", "产品名称"]
WORDS_EN = ["OATS", "Serum", "WATER", "Cotton", "Size", "Model", "Steel", "Fresh", "Light", "Pure", "Organic", "Daily"]


class Recogniser:
    def __init__(self):
        text = (OCR_DIR / "charset_32_94_3944.h").read_text(encoding="utf-8")
        block = text[text.index("CHARSET_CN_3944"):]
        block = block[block.index("{") + 1: block.index("};")]
        self.charset = re.findall(r'u"((?:[^"\\]|\\.)*)"', block)
        self.charset = [c.encode().decode("unicode_escape") if "\\" in c else c for c in self.charset]
        self.net = cv2.dnn.readNet(str(OCR_DIR / "text_recognition_CRNN_CN_2021nov.onnx"))
        self.target = np.array([[0, 31], [0, 0], [99, 0], [99, 31]], np.float32)

    def read(self, rgb: np.ndarray, quad: np.ndarray) -> str:
        m = cv2.getPerspectiveTransform(quad.astype(np.float32), self.target)
        crop = cv2.warpPerspective(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, (100, 32))
        self.net.setInput(cv2.dnn.blobFromImage(crop, size=(100, 32), mean=127.5, scalefactor=1 / 127.5))
        idx = self.net.forward()[:, 0, :].argmax(axis=1)
        chars, prev = [], 0
        for c in idx:
            if c != 0 and c != prev:
                chars.append(self.charset[c - 1])
            prev = c
        return "".join(chars)


def strings_for(seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(12):
        k = i % 4
        if k == 0:
            out.append("".join(str(d) for d in rng.integers(0, 10, 6)))
        elif k == 1:
            out.append(WORDS_CN[int(rng.integers(len(WORDS_CN)))])
        elif k == 2:
            out.append(WORDS_EN[int(rng.integers(len(WORDS_EN)))])
        else:
            out.append(f"{int(rng.integers(10, 999))}{['mL', 'g', 'kg', 'cm'][int(rng.integers(4))]}")
    return out


def render(size: int, strings: list[str], bg) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    im = Image.new("RGB", (HR, HR), bg)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, size)
    items = []
    for i, s in enumerate(strings):
        x = 40 + (i % 3) * (HR - 80) // 3
        y = 60 + (i // 3) * max(size * 2, 200)
        d.text((x, y), s, font=f, fill=[(20, 20, 20), (180, 30, 30), (20, 60, 150)][i % 3])
        l, t, r, b = d.textbbox((x, y), s, font=f)
        p = max(2, size // 6)
        items.append((s, np.array([[l - p, b + p], [l - p, t - p], [r + p, t - p], [r + p, b + p]], np.float32)))
    return np.asarray(im), items


def degrade(hr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "defocus":
        hr = cv2.GaussianBlur(hr, (0, 0), 3.0)
    lr = cv2.resize(hr, (HR // 4, HR // 4), interpolation=cv2.INTER_AREA)
    if kind == "noisy":
        lr = np.clip(lr + np.random.default_rng(0).normal(0, 8, lr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(lr).save(buf, "JPEG", quality=75)
    return np.asarray(Image.open(buf).convert("RGB"))


def main() -> None:
    eng = get_engine()
    ocr = Recogniser()
    det = textguard.get_detector()
    tallies: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])  # ok, wrong, n
    box_heights: dict[float, list[float]] = defaultdict(list)
    for size in SIZES_HR:
        readable = 0
        for c in range(3):
            hr, items = render(size, strings_for(size * 10 + c), BGS[c])
            clean = [(s, q) for s, q in items if ocr.read(hr, q) == s]
            readable += len(clean)
            for kind in KINDS:
                lr = degrade(hr, kind)
                lanczos = np.asarray(Image.fromarray(lr).resize((HR, HR), Image.Resampling.LANCZOS))
                for quad in det.detect(lr):
                    box_heights[size / 4].append(textguard.quad_height(quad))
                safe = textguard.conservative(lr, (HR, HR))
                outs = {"Lanczos": lanczos, "保守": safe}
                for t in TIERS.values():
                    sr = wavelet_color_fix(eng.upscale(lr, t["spec"]), lanczos, device=eng.torch_device)
                    outs[t["label"]] = sr
                    for th in THRESHOLDS:
                        outs[f"{t['label']}+保护{th}"] = textguard.protect(lr, sr, safe, th)[0]
                for method, out in outs.items():
                    tl = tallies[(kind, size / 4, method)]
                    for s, q in clean:
                        r = ocr.read(out, q)
                        tl[2] += 1
                        if r == s:
                            tl[0] += 1
                        elif len(r) >= max(2, len(s) - 1):
                            tl[1] += 1
        print(f"size {size} done ({readable}/36 readable on HR)", file=sys.stderr, flush=True)

    rows = [{"degradation": k, "src_glyph_px": g, "method": m, "n": v[2],
             "read_ok": round(v[0] / v[2], 3) if v[2] else None, "wrong": round(v[1] / v[2], 3) if v[2] else None}
            for (k, g, m), v in tallies.items()]
    heights = {k: (round(float(np.median(v)), 1) if v else None) for k, v in sorted(box_heights.items())}
    (ROOT / "eval" / "report").mkdir(exist_ok=True)
    (ROOT / "eval" / "report" / "text_eval.json").write_text(
        json.dumps({"rows": rows, "detector_box_height_src_px_median": heights}, ensure_ascii=False, indent=2), encoding="utf-8")

    methods = list(dict.fromkeys(m for (_, _, m) in tallies))
    lines = []
    for kind in KINDS + ("ALL",):
        ks = KINDS if kind == "ALL" else (kind,)
        lines.append(f"\n== {kind}  (OCR 读对率 / 读成别的字率)")
        lines.append("glyph_px | " + " | ".join(methods))
        for size in SIZES_HR:
            cells = []
            for m in methods:
                ok = sum(tallies[(k, size / 4, m)][0] for k in ks)
                wr = sum(tallies[(k, size / 4, m)][1] for k in ks)
                n = sum(tallies[(k, size / 4, m)][2] for k in ks)
                cells.append(f"{ok / n:.2f}/{wr / n:.2f}" if n else "-")
            lines.append(f"{size / 4:g} | " + " | ".join(cells))
    lines.append(f"\ndetector box height (src px) by glyph px: {heights}")
    text = "\n".join(lines)
    (ROOT / "eval" / "report" / "text_eval.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
