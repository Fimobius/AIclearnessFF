"""AIClear self-test (ships inside the Windows and macOS packages).

Run with the bundled Python (double-click "Run Self-Test.command" on macOS, or
`runtime\\python.exe selftest.py` on Windows). It:
  1. checks that LAN mode refuses to start without an access key,
  2. starts the app through the real start script with a stripped-down environment,
  3. processes generated test images on both enhancement tiers and verifies exact output
     size, text protection, the ZIP download and the "already running" detection,
  4. writes selftest_report.txt (system info, backend, accelerator check, timings, log).
Exit code 0 = passed.
"""
from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = sys.platform == "darwin"
REPORT = HERE / "selftest_report.txt"
lines: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def http(method: str, url: str, data: bytes | None = None, headers: dict | None = None, timeout=600):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
    """The launcher records its port in <data>/server.json (probing a port range is slow on
    Windows: ~1 s per closed port)."""
    sys.path.insert(0, str(HERE))
    import launcher
    port = launcher.recorded_port(launcher.data_dir())
    if port and launcher.running_instance(port):
        return f"http://127.0.0.1:{port}"
    return None


def test_images() -> list[tuple[str, bytes]]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def jpeg(im: Image.Image, q=75) -> bytes:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        return buf.getvalue()

    # product label: small text that becomes ~7 px tall after the x4 shrink -> must be protected
    label = Image.new("RGB", (1024, 512), "white")
    d = ImageDraw.Draw(label)
    font = ImageFont.load_default(size=28)
    d.rectangle((20, 20, 1004, 492), outline=(200, 30, 30), width=8)
    d.text((60, 60), "NET WT 500g  6901234567892", font=font, fill=(20, 20, 20))
    d.text((60, 140), "BEST BEFORE 2026-11-08  LOT 83A", font=font, fill=(150, 20, 20))
    d.text((60, 260), "ORGANIC OATS", font=ImageFont.load_default(size=96), fill=(20, 60, 150))
    label = label.resize((256, 128), Image.Resampling.LANCZOS)
    # photo-like: gradients, shapes and noise
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    photo = np.stack([120 + 80 * np.sin(xx / 23), 100 + 60 * np.cos(yy / 31), 140 + 50 * np.sin((xx + yy) / 40)], -1)
    photo = np.clip(photo + rng.normal(0, 8, photo.shape), 0, 255).astype(np.uint8)
    return [("商品 标签.jpg", jpeg(label)), ("风景.jpg", jpeg(Image.fromarray(photo))), ("坏文件.jpg", b"not an image")]


def stripped_env() -> dict[str, str]:
    if MAC:
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": os.environ["HOME"],
               "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "LANG": "en_US.UTF-8"}
    else:
        win = os.environ["SystemRoot"]
        env = {"SystemRoot": win, "PATH": f"{win}\\System32;{win}", "TEMP": os.environ["TEMP"],
               "TMP": os.environ["TEMP"], "LOCALAPPDATA": os.environ["LOCALAPPDATA"],
               "USERPROFILE": os.environ["USERPROFILE"], "COMSPEC": os.environ["COMSPEC"]}
    env["AICLEAR_NO_BROWSER"] = "1"
    for k in ("AICLEAR_DEVICE",):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def start_cmd() -> list[str]:
    if MAC:
        return ["/bin/bash", str(HERE / "Start AIClear.command")]
    return ["cmd", "/c", str(HERE / "Start AIClear.bat")]


def system_info() -> None:
    say(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"os: {platform.platform()}  mac_ver={platform.mac_ver()[0] or '-'}  machine={platform.machine()}")
    say(f"python: {sys.version.split()[0]}  package dir: {HERE}")
    try:
        import onnxruntime as ort
        import cv2
        import numpy
        say(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}  "
            f"opencv {cv2.__version__}  numpy {numpy.__version__}")
    except Exception as e:
        say(f"import check FAILED: {e}")
        raise


def run() -> None:
    system_info()
    assert find_server() is None, "AIClear 已经在运行，请先关闭它再自检"
    env = stripped_env()

    # 1. LAN mode without an access key must refuse to start
    r = subprocess.run(start_cmd(), cwd=HERE, env={**env, "AICLEAR_HOST": "0.0.0.0"}, input=b"\n\r\n",
                       capture_output=True, timeout=180)
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    assert "AICLEAR_API_KEY" in out and find_server() is None, "局域网模式没有拦截：\n" + out
    say("[ok] LAN mode without access key is refused")

    # 2. start the app
    log_path = HERE / "selftest_server.log"
    log = open(log_path, "wb")
    kw = {} if MAC else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    proc = subprocess.Popen(start_cmd(), cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kw)
    try:
        t0 = time.time()
        base = None
        say("starting server (the first start after unzipping can take minutes: security scanning) ...")
        while time.time() - t0 < 600 and not base:
            base = find_server()
            time.sleep(0.5)
        assert base, "服务 10 分钟内没有启动"
        say(f"[ok] server up in {time.time() - t0:.1f}s at {base}")
        h = http("GET", f"{base}/api/health")
        assert all(h["models"][t["model"]]["available"] for t in h["tiers"].values()), "模型文件缺失"
        assert h["text_protect_available"], "文字检测模型缺失"
        assert b'<div id="root">' in http("GET", f"{base}/"), "网页界面缺失"
        say("[ok] models and web UI present")

        # 3. process on both tiers
        imgs = test_images()
        for tier, files in (("strong", imgs), ("standard", imgs[1:2])):
            settings = {"tier": tier, "width": 1000, "ratio_w": 10, "ratio_h": 13, "mode": "pad", "format": "jpg"}
            body, ctype = multipart(files, {"start": "true", "settings": json.dumps(settings)})
            t1 = time.time()
            job = http("POST", f"{base}/api/jobs", body, {"Content-Type": ctype})
            while job["status"] in ("queued", "running"):
                time.sleep(1)
                job = http("GET", f"{base}/api/jobs/{job['id']}")
            done = [i for i in job["items"] if i["status"] == "done"]
            failed = [i for i in job["items"] if i["status"] == "failed"]
            assert not failed, f"{tier}: 处理失败 {[i['error'] for i in failed]}"
            assert len(done) == len([f for f in files if f[1] != b"not an image"]), job["counts"]
            from PIL import Image
            for it in done:
                im = Image.open(io.BytesIO(http("GET", base + it["urls"]["result"])))
                im.load()
                assert im.size == (1000, 1300), f"输出尺寸错误 {im.size}"
                say(f"[ok] {tier}: {it['original_name']} -> {im.size} in {it['result']['seconds']}s "
                    f"(ssim {it['result']['qa']['ssim']}, dE {it['result']['qa']['delta_e_mean']}, "
                    f"text protected {it['result']['text']['protected']})")
            if tier == "strong":
                assert job["counts"]["invalid"] == 1, "坏文件没有被拦截"
                label = next(i for i in done if i["original_name"].startswith("商品"))
                assert label["result"]["text"]["protected"] >= 1, "小文字保护没有生效"
                z = zipfile.ZipFile(io.BytesIO(http("GET", f"{base}/api/jobs/{job['id']}/download")))
                assert len(z.namelist()) == 3, z.namelist()
                say(f"[ok] broken file rejected, text protection active, zip ok ({time.time() - t1:.1f}s)")

        dev = http("GET", f"{base}/api/health")["device"]
        say(f"backend: {dev.get('backend')} ({dev.get('device')})")
        for model, chk in (dev.get("accel_checks") or {}).items():
            say(f"  accelerator check {model}: {chk}")

        # 4. second launch must detect the running instance
        r = subprocess.run(start_cmd(), cwd=HERE, env=env, capture_output=True, timeout=120, input=b"\n")
        assert "已在运行" in r.stdout.decode("utf-8", errors="replace"), r.stdout
        say("[ok] second launch detects the running instance")
    finally:
        if MAC:
            subprocess.run(["pkill", "-f", str(HERE / "launcher.py")], capture_output=True)
            proc.terminate()
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        log.close()


def main() -> int:
    ok = False
    try:
        run()
        ok = True
        say("\n自检通过 SELFTEST PASSED")
    except Exception:
        say(traceback.format_exc())
        say("\n自检失败 SELFTEST FAILED")
    log_path = HERE / "selftest_server.log"
    if log_path.exists():
        lines.append("\n---- server log (tail) ----")
        lines.append(log_path.read_text(encoding="utf-8", errors="replace")[-6000:])
    try:
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"报告已保存：{REPORT}")
    except OSError as e:
        print(f"无法写报告：{e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
