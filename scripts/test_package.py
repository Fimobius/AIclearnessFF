"""Smoke-test an extracted release package in a stripped-down environment.

Starts `Start AIClear.bat` via cmd with only C:\\Windows on PATH (no Python, no venv, no
PyTorch visible), then drives the HTTP API: health, upload (incl. a broken file), process,
verify output sizes / text protection / ZIP, the "already running" detection, and that LAN
mode refuses to start without an access key.

Usage: backend/.venv/Scripts/python scripts/test_package.py <extracted AIClear dir> [--cpu]
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def http(method: str, url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
        return json.loads(body) if r.headers.get("content-type", "").startswith("application/json") else body


def multipart(files: list[tuple[str, bytes]], fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----aiclear" + str(time.time_ns())
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for name, data in files:
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n".encode())
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def find_server() -> str | None:
    for port in range(8000, 8010):
        try:
            if http("GET", f"http://127.0.0.1:{port}/api/health").get("app") == "AIClear":
                return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return None


def main() -> None:
    pkg = Path(sys.argv[1]).resolve()
    cpu = "--cpu" in sys.argv
    win = os.environ["SystemRoot"]
    env = {"SystemRoot": win, "PATH": f"{win}\\System32;{win}", "TEMP": os.environ["TEMP"], "TMP": os.environ["TEMP"],
           "LOCALAPPDATA": os.environ["LOCALAPPDATA"], "USERPROFILE": os.environ["USERPROFILE"],
           "COMSPEC": os.environ["COMSPEC"], "AICLEAR_NO_BROWSER": "1"}
    if cpu:
        env["AICLEAR_DEVICE"] = "cpu"
    bat = str(pkg / "Start AIClear.bat")
    assert find_server() is None, "another AIClear instance is already running; stop it first"

    # LAN mode without an access key must refuse to start (input answers the bat's "pause")
    r = subprocess.run(["cmd", "/c", bat], cwd=pkg, env={**env, "AICLEAR_HOST": "0.0.0.0"},
                       input=b"\r\n", capture_output=True, timeout=120)
    out = r.stdout.decode("utf-8", errors="replace") + r.stderr.decode("utf-8", errors="replace")
    assert "AICLEAR_API_KEY" in out and find_server() is None, out
    print("LAN without access key: refused")

    log = open(pkg / "test_server.log", "wb")
    proc = subprocess.Popen(["cmd", "/c", bat], cwd=pkg, env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        base = None
        for _ in range(240):
            base = find_server()
            if base:
                break
            time.sleep(0.5)
        assert base, "server did not start:\n" + (pkg / "test_server.log").read_text(encoding="utf-8", errors="replace")
        h = http("GET", f"{base}/api/health")
        print("health:", h["device"]["backend"], {k: v["model"] for k, v in h["tiers"].items()})
        assert all(h["models"][t["model"]]["available"] for t in h["tiers"].values())
        assert h["text_protect_available"]
        assert b'<div id="root">' in http("GET", f"{base}/"), "web UI not served"

        pairs = ROOT / "eval" / "golden" / "pairs"
        files = [("商品 标签.png", (pairs / "prod_box_text" / "compressed.png").read_bytes()),
                 ("风景.png", (pairs / "div2k_0039" / "noisy.png").read_bytes()),
                 ("坏文件.jpg", b"not an image")]
        settings = {"tier": "strong", "width": 1000, "ratio_w": 10, "ratio_h": 13, "mode": "pad", "format": "jpg"}
        body, ctype = multipart(files, {"start": "true", "settings": json.dumps(settings)})
        t0 = time.time()
        job = http("POST", f"{base}/api/jobs", body, {"Content-Type": ctype})
        while job["status"] in ("queued", "running"):
            time.sleep(1)
            job = http("GET", f"{base}/api/jobs/{job['id']}")
        print(f"job: {job['status']} in {time.time() - t0:.1f}s, counts={ {k: v for k, v in job['counts'].items() if v} }")
        assert job["counts"]["done"] == 2 and job["counts"]["invalid"] == 1, job
        from PIL import Image
        for it in job["items"]:
            if it["status"] != "done":
                continue
            im = Image.open(io.BytesIO(http("GET", base + it["urls"]["result"])))
            assert im.size == (1000, 1300), im.size
            print(f"  {it['result']['seconds']}s text_protected={it['result']['text']['protected']} "
                  f"ssim={it['result']['qa']['ssim']} dE={it['result']['qa']['delta_e_mean']}")
        label = next(i for i in job["items"] if i["original_name"].startswith("商品"))
        assert label["result"]["text"]["protected"] >= 1, "text protection did not trigger"
        z = zipfile.ZipFile(io.BytesIO(http("GET", f"{base}/api/jobs/{job['id']}/download")))
        assert len(z.namelist()) == 3 and all(i.flag_bits & 0x800 for i in z.infolist() if not i.filename.isascii())
        print("zip: 2 images + report, UTF-8 names")

        # a second launch must detect the running instance and exit immediately
        r = subprocess.run(["cmd", "/c", bat], cwd=pkg, env=env, capture_output=True, timeout=60)
        assert "已在运行" in r.stdout.decode("utf-8", errors="replace"), r.stdout
        print("second launch: detected running instance")
        print("PACKAGE TEST PASSED", "(CPU)" if cpu else "")
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


if __name__ == "__main__":
    main()
