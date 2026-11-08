"""Build the portable Windows package: build/AIClear-v<version>-Windows-x64.zip

Layout of the package:
  Start AIClear.bat    double-click to start
  ReadMe.txt           user guide (Chinese)
  launcher.py          picks a port, starts the server, opens the browser
  runtime/             relocatable CPython 3.12 (python-build-standalone) + dependencies
                       + MSVC runtime DLLs (app-local, so no VC++ redistributable is needed)
  app_src/app/         the backend
  web/                 the built web UI
  models/              ONNX models (SwinIR x2, PP-OCRv3 text detector) + licences

Prerequisites (dev machine): ONNX models exported (scripts/export_onnx.py), frontend built
(npm run build), uv available (`python -m uv`).

Usage: backend/.venv/Scripts/python scripts/build_release.py
"""
from __future__ import annotations

import compileall
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PKG = BUILD / "AIClear"
PYTHON_HOME = Path(os.environ.get("AICLEAR_BUILD_PYTHON",
                                  Path(os.environ["APPDATA"]) / "uv" / "python" / "cpython-3.12.16-windows-x86_64-none"))
INDEX = os.environ.get("PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
VC_DLLS = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"]
MODEL_FILES = ["SwinIR-M_x4_PSNR_t256.onnx", "SwinIR-M_x4_GAN_t256.onnx", "text_detection_cn_ppocrv3_2023may.onnx",
               "MODELS.md", "LICENSE_text_detection_ppocr.txt"]
RUNTIME_SKIP = {"tcl", "include", "libs", "Scripts"}
LIB_SKIP = {"test", "tkinter", "idlelib", "turtledemo", "ensurepip", "lib2to3", "pydoc_data"}


def version() -> str:
    m = re.search(r'VERSION = "([^"]+)"', (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8"))
    return m.group(1)


def copytree(src: Path, dst: Path, skip=()) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", *skip), dirs_exist_ok=True)


def crlf(src: Path, dst: Path, bom: bool = False) -> None:
    text = src.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
    dst.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))


def main() -> None:
    ver = version()
    for f in MODEL_FILES:
        if not (ROOT / "models" / f).exists():
            sys.exit(f"missing models/{f} - run scripts/export_onnx.py first")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        sys.exit("frontend not built - run npm run build in frontend/")

    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    print("runtime ...")
    rt = PKG / "runtime"
    rt.mkdir()
    for item in PYTHON_HOME.iterdir():
        if item.name in RUNTIME_SKIP:
            continue
        if item.name == "Lib":
            copytree(item, rt / "Lib", skip=[*LIB_SKIP, "site-packages"])
            (rt / "Lib" / "site-packages").mkdir()
        elif item.is_dir():
            copytree(item, rt / item.name)
        else:
            shutil.copy2(item, rt / item.name)
    for dll in VC_DLLS:
        shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / dll, rt / dll)

    print("dependencies ...")
    # uv: AICLEAR_UV, or on PATH (set AICLEAR_UV when it only lives in another Python's Scripts dir)
    uv = [os.environ.get("AICLEAR_UV") or shutil.which("uv") or sys.exit("uv not found; set AICLEAR_UV")]
    subprocess.run([*uv, "pip", "install",
                    "--target", str(rt / "Lib" / "site-packages"),
                    "--python-version", "3.12", "--python-platform", "x86_64-pc-windows-msvc",
                    "-r", str(ROOT / "backend" / "requirements-runtime.txt"), "--index-url", INDEX], check=True)

    print("app, web, models ...")
    copytree(ROOT / "backend" / "app", PKG / "app_src" / "app")
    copytree(ROOT / "frontend" / "dist", PKG / "web")
    (PKG / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy2(ROOT / "models" / f, PKG / "models" / f)
    shutil.copy2(ROOT / "packaging" / "launcher.py", PKG / "launcher.py")
    # ASCII file names: older Explorer / unzip tools ignore the zip UTF-8 flag
    crlf(ROOT / "packaging" / "Start AIClear.bat", PKG / "Start AIClear.bat")
    crlf(ROOT / "packaging" / "ReadMe.txt", PKG / "ReadMe.txt", bom=True)
    crlf(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt", PKG / "THIRD_PARTY_NOTICES.txt")

    print("byte-compiling ...")
    compileall.compile_dir(str(PKG), quiet=1, workers=0, legacy=False)

    size = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file())
    print(f"package folder: {size / 1e6:.0f} MB")
    zpath = BUILD / f"AIClear-v{ver}-Windows-x64.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(PKG.rglob("*")):
            if f.is_file():
                z.write(f, Path("AIClear") / f.relative_to(PKG))
    print(f"zip: {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
