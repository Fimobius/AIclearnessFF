"""Launcher for the portable Windows package.

- Keeps data next to the program if that folder is writable, else in %LOCALAPPDATA%\\AIClear.
- If the app is already running, just opens the browser.
- Otherwise picks a free port (8000-8099), starts the server and opens the browser when ready.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_ID = "AIClear"


def writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def running_instance(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
            return json.load(r).get("app") == APP_ID
    except Exception:
        return False


def free_port(start: int = 8000) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("找不到可用端口（8000-8099 都被占用）")


def main() -> None:
    data = HERE / "data"
    if not writable(data):
        data = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_ID / "data"
        data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AICLEAR_DATA_DIR", str(data))
    os.environ.setdefault("AICLEAR_MODELS_DIR", str(HERE / "models"))
    os.environ.setdefault("AICLEAR_FRONTEND_DIST", str(HERE / "web"))
    os.environ.setdefault("AICLEAR_BACKEND", "onnx")
    sys.path.insert(0, str(HERE / "app_src"))

    for port in range(8000, 8100):
        if running_instance(port):
            print(f"程序已在运行，正在打开浏览器：http://127.0.0.1:{port}")
            webbrowser.open(f"http://127.0.0.1:{port}")
            return

    # LAN sharing (e.g. Mac colleagues using this PC): AICLEAR_HOST=0.0.0.0 + AICLEAR_API_KEY
    host = os.environ.get("AICLEAR_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost") and not os.environ.get("AICLEAR_API_KEY"):
        raise SystemExit("开放到局域网（AICLEAR_HOST）时必须同时设置访问密钥 AICLEAR_API_KEY")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("  AI 图片高清化 & 尺寸标准化")
    print(f"  浏览器地址：{url}")
    print(f"  数据目录：{data}")
    if host not in ("127.0.0.1", "localhost"):
        try:
            ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
        except OSError:
            ips = []
        for ip in ips:
            print(f"  局域网地址：http://{ip}:{port}（需要访问密钥）")
    print("  关闭这个窗口即退出程序")
    print("=" * 60, flush=True)

    def open_when_ready() -> None:
        for _ in range(240):
            if running_instance(port):
                if not os.environ.get("AICLEAR_NO_BROWSER"):
                    webbrowser.open(url)
                return
            time.sleep(0.5)
        print("服务启动超时，请查看上方的错误信息", flush=True)

    threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        print()
        print("程序启动失败。请把上面的信息截图发给技术支持。", flush=True)
        sys.exit(1)
