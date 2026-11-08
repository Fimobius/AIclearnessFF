"""Model bake-off on the golden set (x4 task).

For every pair in golden/pairs/<name>/{hr,<degradation>}.png, each candidate upscales the
LR image x4 (with the same colour lock as production), and is scored against HR:
  PSNR / SSIM (Y channel)  - fidelity
  LPIPS (AlexNet)          - perceptual distance, lower is better
  ΔE2000 (mean)            - colour accuracy, < 2.3 is below the just-noticeable difference
Writes report/results.csv, report/summary.json and report/index.html (metrics + visual crops).

Usage: python eval/bakeoff.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.imaging.colorfix import wavelet_color_fix  # noqa: E402
from app.imaging.sr import ModelSpec, get_engine  # noqa: E402

HERE = Path(__file__).parent
PAIRS = HERE / "golden" / "pairs"
REPORT = HERE / "report"

CANDIDATES: list[tuple[str, ModelSpec | None, bool]] = [
    ("Lanczos（基线，无 AI）", None, False),
    ("RealESRNet_x4plus", ModelSpec("RealESRNet_x4plus"), True),
    ("RealESRGAN_x4plus", ModelSpec("RealESRGAN_x4plus"), True),
    ("RealESRGAN_x4plus（无色彩保护）", ModelSpec("RealESRGAN_x4plus"), False),
    ("general-x4v3 降噪0.3", ModelSpec("realesr-general-x4v3", 0.3), True),
    ("general-x4v3 降噪0.5", ModelSpec("realesr-general-x4v3", 0.5), True),
    ("general-x4v3 降噪0.8", ModelSpec("realesr-general-x4v3", 0.8), True),
    ("SwinIR-M x4 PSNR", ModelSpec("SwinIR-M_x4_PSNR"), True),
    ("SwinIR-M x4 GAN", ModelSpec("SwinIR-M_x4_GAN"), True),
]


def y_channel(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[..., 0]


class Lpips:
    def __init__(self):
        import lpips
        self.net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    @torch.inference_mode()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1  # noqa: E731
        return float(self.net(t(a), t(b)))


def run(limit: int | None) -> None:
    eng = get_engine()
    lp = Lpips()
    REPORT.mkdir(exist_ok=True)
    (REPORT / "crops").mkdir(exist_ok=True)
    pairs = sorted(p for p in PAIRS.iterdir() if p.is_dir())[: limit or None]
    rows = []
    crops: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    # warm-up so the first timing isn't CUDA init
    eng.upscale(np.zeros((64, 64, 3), np.uint8), ModelSpec("RealESRGAN_x4plus"))
    for pair in pairs:
        hr = np.asarray(Image.open(pair / "hr.png").convert("RGB"))
        H, W = hr.shape[:2]
        # Crop window for the visual comparison: the most detailed 256px patch
        g = cv2.Laplacian(y_channel(hr), cv2.CV_32F) ** 2
        g = cv2.boxFilter(g, -1, (256, 256))[128:H - 128, 128:W - 128]
        cy, cx = np.unravel_index(np.argmax(g), g.shape)
        win = (cx, cy, cx + 256, cy + 256)
        Image.fromarray(hr[win[1]:win[3], win[0]:win[2]]).save(REPORT / "crops" / f"{pair.name}__hr.png")
        for deg in sorted(p.stem for p in pair.glob("*.png") if p.stem != "hr"):
            lr = np.asarray(Image.open(pair / f"{deg}.png").convert("RGB"))
            lanczos = np.asarray(Image.fromarray(lr).resize((W, H), Image.Resampling.LANCZOS))
            Image.fromarray(lr).resize((W, H), Image.Resampling.NEAREST).crop(win).save(
                REPORT / "crops" / f"{pair.name}__{deg}__input.png")
            for ci, (label, spec, fix) in enumerate(CANDIDATES):
                t0 = time.perf_counter()
                if spec is None:
                    out = lanczos
                else:
                    out = eng.upscale(lr, spec)
                    if fix:
                        out = wavelet_color_fix(out, lanczos, device=eng.torch_device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                de = deltaE_ciede2000(rgb2lab(hr), rgb2lab(out))
                row = {
                    "image": pair.name, "degradation": deg, "method": label,
                    "psnr": peak_signal_noise_ratio(y_channel(hr), y_channel(out), data_range=255),
                    "ssim": structural_similarity(y_channel(hr), y_channel(out), data_range=255),
                    "lpips": lp(hr, out), "delta_e": float(de.mean()), "seconds": dt,
                }
                rows.append(row)
                fn = f"{pair.name}__{deg}__m{ci}.png"
                Image.fromarray(out[win[1]:win[3], win[0]:win[2]]).save(REPORT / "crops" / fn)
                crops[(pair.name, deg)][label] = fn
            print(f"{pair.name}/{deg} done", file=sys.stderr, flush=True)

    with open(REPORT / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(filter_fn) -> dict[str, dict[str, float]]:
        acc: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if filter_fn(r):
                acc[r["method"]].append(r)
        return {m: {k: float(np.mean([x[k] for x in v])) for k in ("psnr", "ssim", "lpips", "delta_e", "seconds")}
                for m, v in acc.items()}

    degs = sorted({r["degradation"] for r in rows})
    summary = {"overall": agg(lambda r: True),
               "products": agg(lambda r: r["image"].startswith("prod_")),
               **{f"deg:{d}": agg(lambda r, d=d: r["degradation"] == d) for d in degs},
               "n_pairs": len(pairs), "n_cases": len(rows) // len(CANDIDATES)}
    (REPORT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(summary, crops, degs)
    print(f"report -> {REPORT / 'index.html'}", file=sys.stderr)


def write_html(summary: dict, crops: dict, degs: list[str]) -> None:
    labels = [c[0] for c in CANDIDATES]

    def table(title: str, data: dict) -> str:
        best = {k: (min if k in ("lpips", "delta_e", "seconds") else max)(data[m][k] for m in data)
                for k in ("psnr", "ssim", "lpips", "delta_e", "seconds")}
        trs = []
        for m in labels:
            if m not in data:
                continue
            tds = "".join(
                f'<td class="{"best" if abs(data[m][k] - best[k]) < 1e-9 else ""}">{data[m][k]:.{p}f}</td>'
                for k, p in (("psnr", 2), ("ssim", 4), ("lpips", 4), ("delta_e", 2), ("seconds", 3)))
            trs.append(f"<tr><th>{html.escape(m)}</th>{tds}</tr>")
        return (f"<h3>{html.escape(title)}</h3><table><tr><th>方法</th><th>PSNR↑</th><th>SSIM↑</th>"
                f"<th>LPIPS↓</th><th>ΔE2000↓</th><th>秒/张↓</th></tr>{''.join(trs)}</table>")

    names = {"overall": "全部样本", "products": "仅商品图（含文字/Logo/色卡）",
             **{f"deg:{d}": f"退化类型：{d}" for d in degs}}
    parts = [table(names[k], v) for k, v in summary.items() if k in names]
    grids = []
    for (img, deg), m in sorted(crops.items()):
        cells = [f'<figure><img src="crops/{img}__{deg}__input.png"><figcaption>输入（放大显示）</figcaption></figure>']
        cells += [f'<figure><img src="crops/{m[l]}"><figcaption>{html.escape(l)}</figcaption></figure>' for l in labels if l in m]
        cells.append(f'<figure><img src="crops/{img}__hr.png"><figcaption>原始高清（参考答案）</figcaption></figure>')
        grids.append(f"<h3>{img} · {deg}</h3><div class=grid>{''.join(cells)}</div>")
    doc = f"""<!doctype html><meta charset=utf-8><title>模型对比报告</title>
<style>body{{font-family:system-ui,"Microsoft YaHei";margin:24px;color:#222}}table{{border-collapse:collapse;margin:8px 0 24px}}
td,th{{border:1px solid #ddd;padding:4px 10px;text-align:right}}th{{text-align:left}}td.best{{background:#d9f5dd;font-weight:600}}
.grid{{display:flex;flex-wrap:wrap;gap:6px}}figure{{margin:0;width:200px}}figure img{{width:200px;height:200px;image-rendering:pixelated}}
figcaption{{font-size:12px;color:#555}}</style>
<h1>超分模型对比报告（x4，{summary['n_pairs']} 张图 × 退化 = {summary['n_cases']} 个样本）</h1>
<p>指标以原始高清图为参考答案。PSNR/SSIM 衡量保真，LPIPS 衡量感知差距（越低越接近原图观感），ΔE2000 衡量色差（&lt;2.3 人眼难以察觉）。绿色为该列最优。</p>
{''.join(parts)}<h2>视觉对比（每张图细节最丰富的 256px 区域）</h2>{''.join(grids)}"""
    (REPORT / "index.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    run(ap.parse_args().limit)
