"""Stability acceptance test (PRD: "批量几十/上百张图片场景下，不会崩溃、中断任务").

1. Starts its own server on a temp data dir.
2. Uploads 200 files of mixed size/format, including 10 broken ones (5%) and edge cases
   (CMYK, transparent PNG, 16-bit PNG, EXIF-rotated JPEG).
3. Hard-kills the server while the batch is running, restarts it, and checks that the
   queue resumes without losing or duplicating any image.
4. Verifies every output opens and is exactly the configured size, and the ZIP is complete.

Usage: backend/.venv/Scripts/python eval/stress_test.py [--n 200]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
SETTINGS = {"tier": "standard", "width": 800, "ratio_w": 10, "ratio_h": 10, "mode": "pad", "format": "jpg", "quality": 90}


def make_files(n: int, rng: np.random.Generator) -> list[tuple[str, bytes, str]]:
    """Returns (name, data, kind) with kind in {ok, broken}."""
    hrs = sorted((ROOT / "eval" / "golden" / "pairs").glob("*/hr.png"))
    srcs = [Image.open(p).convert("RGB") for p in hrs]
    files: list[tuple[str, bytes, str]] = []
    n_broken = max(1, n // 20)
    for i in range(n - n_broken - 4):
        im = srcs[i % len(srcs)]
        w = int(rng.integers(160, 1600))
        h = int(w * rng.uniform(0.6, 1.6))
        x, y = int(rng.integers(0, 200)), int(rng.integers(0, 200))
        crop = im.crop((x, y, x + 800, y + 800)).resize((w, h), Image.Resampling.BICUBIC)
        fmt = ["JPEG", "PNG", "WEBP"][i % 3]
        buf = io.BytesIO()
        crop.save(buf, fmt, **({"quality": int(rng.integers(40, 90))} if fmt != "PNG" else {}))
        files.append((f"商品_{i:03d}.{fmt.lower().replace('jpeg', 'jpg')}", buf.getvalue(), "ok"))
    # edge cases
    base = srcs[0].resize((400, 300))
    buf = io.BytesIO(); base.convert("CMYK").save(buf, "JPEG"); files.append(("edge_cmyk.jpg", buf.getvalue(), "ok"))
    rgba = base.convert("RGBA"); a = np.array(rgba); a[:50, :, 3] = 0
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG"); files.append(("edge_alpha.png", buf.getvalue(), "ok"))
    buf = io.BytesIO(); Image.fromarray((np.asarray(base.convert("L"), np.uint16) * 257)).save(buf, "PNG")
    files.append(("edge_16bit.png", buf.getvalue(), "ok"))
    ex = base.getexif(); ex[0x0112] = 6
    buf = io.BytesIO(); base.save(buf, "JPEG", exif=ex.tobytes()); files.append(("edge_exif_rotated.jpg", buf.getvalue(), "ok"))
    # broken
    good = files[0][1]
    broken = [good[: len(good) // 3], b"", b"this is not an image", good[:100], os.urandom(5000)]
    for j in range(n_broken):
        files.append((f"broken_{j}.jpg", broken[j % len(broken)], "broken"))
    order = rng.permutation(len(files))
    return [files[k] for k in order]


def kill_tree(p: subprocess.Popen) -> None:
    """Hard-kill the server. On Windows the venv python.exe is a launcher that spawns the
    real interpreter as a child, so the whole process tree must go."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        p.kill()
    p.wait()
    for _ in range(40):  # wait until the port is really free
        try:
            httpx.get(f"{BASE}/api/health", timeout=0.5)
            time.sleep(0.25)
        except httpx.HTTPError:
            return


def start_server(data_dir: str) -> subprocess.Popen:
    env = {**os.environ, "AICLEAR_DATA_DIR": data_dir, "AICLEAR_RETENTION_DAYS": "0", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([str(PY), "-m", "uvicorn", "app.main:app", "--app-dir", str(ROOT / "backend"),
                          "--port", str(PORT), "--log-level", "warning"], env=env,
                         stdout=subprocess.DEVNULL, stderr=open(Path(data_dir) / "server.log", "ab"))
    for _ in range(120):
        try:
            if httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kill-at", type=float, default=0.35, help="kill the server after this fraction is done")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    data_dir = tempfile.mkdtemp(prefix="aiclear-stress-")
    files = make_files(args.n, rng)
    expected_ok = sum(1 for f in files if f[2] == "ok")
    print(f"{len(files)} files ({expected_ok} valid, {len(files) - expected_ok} broken), data dir {data_dir}")

    server = start_server(data_dir)
    c = httpx.Client(base_url=BASE, timeout=120)
    t0 = time.time()
    job = None
    for i in range(0, len(files), 20):
        chunk = [("files", (n, d, "application/octet-stream")) for n, d, _ in files[i:i + 20]]
        job = c.post("/api/jobs" if job is None else f"/api/jobs/{job['id']}/files", files=chunk).json()
    jid = job["id"]
    j = c.get(f"/api/jobs/{jid}").json()
    invalid = j["counts"]["invalid"]
    print(f"uploaded in {time.time() - t0:.1f}s: invalid={invalid}")
    c.post(f"/api/jobs/{jid}/start", json={"settings": SETTINGS}).raise_for_status()
    t_start = time.time()

    killed_at = None
    while True:
        j = c.get(f"/api/jobs/{jid}").json()
        done = j["counts"]["done"] + j["counts"]["failed"]
        if killed_at is None and done >= expected_ok * args.kill_at:
            killed_at = done
            kill_tree(server)  # hard kill, no graceful shutdown
            print(f"killed server at {done} finished; restarting")
            server = start_server(data_dir)
            continue
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(1)
    elapsed = time.time() - t_start

    items = j["items"]
    bad_size, unreadable = [], []
    for it in items:
        if it["status"] != "done":
            continue
        data = c.get(it["urls"]["result"]).content
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            if im.size != (800, 800):
                bad_size.append((it["original_name"], im.size))
        except Exception as e:
            unreadable.append((it["original_name"], str(e)))
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{jid}/download").content))
    zip_imgs = [n for n in z.namelist() if not n.endswith(".csv")]
    reprocessed = [it["original_name"] for it in items if it["attempts"] > 1]
    result = {
        "files": len(files), "valid": expected_ok, "invalid_detected": invalid,
        "done": j["counts"]["done"], "failed": j["counts"]["failed"],
        "failed_items": [(it["original_name"], it["error"]) for it in items if it["status"] == "failed"],
        "wrong_size": bad_size, "unreadable": unreadable, "zip_images": len(zip_imgs),
        "zip_names_unique": len(set(zip_imgs)) == len(zip_imgs),
        "killed_after": killed_at, "reprocessed_after_restart": reprocessed,
        "processing_seconds": round(elapsed, 1), "seconds_per_image": round(elapsed / max(1, j["counts"]["done"]), 2),
        "review_flagged": j["needs_review"],
    }
    kill_tree(server)
    passed = (invalid == len(files) - expected_ok and result["done"] == expected_ok and not bad_size
              and not unreadable and result["zip_images"] == expected_ok and result["zip_names_unique"])
    result["PASSED"] = passed
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (ROOT / "eval" / "report").mkdir(exist_ok=True)
    (ROOT / "eval" / "report" / "stress.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
