"""Build the portable packages.

  backend/.venv/Scripts/python scripts/build_release.py [windows-x64] [macos-arm64] [macos-x64]
  (no argument = all three)  ->  build/AIClear-v<version>-<Platform>.zip

Package layout (all platforms):
  Start AIClear.bat / Start AIClear.command   double-click to start
  Run Self-Test.command (macOS)               double-click to verify the install
  ReadMe.txt                                  user guide (Chinese)
  launcher.py, selftest.py
  runtime/       relocatable CPython 3.12 (python-build-standalone) + dependencies
                 (Windows: + app-local MSVC runtime DLLs)
  app_src/app/   the backend
  web/           the built web UI
  models/        ONNX models (SwinIR x2, PP-OCRv3 text detector) + licences

The macOS packages are assembled on any OS: the Python runtime is the official
python-build-standalone tarball (SHA-256 pinned below) and dependencies are macOS wheels
fetched by uv for macOS 13. Every Mach-O binary is then checked to require at most
macOS 13.4 (scripts/macho_minos.py).

Prerequisites: ONNX models exported (scripts/export_onnx.py), frontend built (npm run build),
uv available (on PATH or AICLEAR_UV).
"""
from __future__ import annotations

import compileall
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
INDEX = os.environ.get("PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
MODEL_FILES = ["SwinIR-M_x4_PSNR_t256.onnx", "SwinIR-M_x4_GAN_t256.onnx", "text_detection_cn_ppocrv3_2023may.onnx",
               "MODELS.md", "LICENSE_text_detection_ppocr.txt"]
LIB_SKIP = {"test", "tkinter", "idlelib", "turtledemo", "ensurepip", "lib2to3", "pydoc_data", "site-packages"}
# Wheels are tagged macosx_13_0, but ONNX Runtime 1.23's binaries actually require 13.4
# (1.20-1.22: 13.3) - measured with scripts/macho_minos.py. Ventura updates to 13.7 for free.
MAC_MIN = "13.4"

WIN_PYTHON = Path(os.environ.get("AICLEAR_BUILD_PYTHON",
                                 Path(os.environ.get("APPDATA", "")) / "uv" / "python" / "cpython-3.12.16-windows-x86_64-none"))
VC_DLLS = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"]

PBS = "https://releases.astral.sh/github/python-build-standalone/releases/download/20261026/"
MAC_PYTHON = {
    "macos-arm64": ("cpython-3.12.16+20261026-aarch64-apple-darwin-install_only_stripped.tar.gz",
                    "353d5f1b4fd2715e7053f12f068435d5bbad16b91e057006b5d01e0925709a7e", "aarch64-apple-darwin"),
    "macos-x64": ("cpython-3.12.16+20261026-x86_64-apple-darwin-install_only_stripped.tar.gz",
                  "37c38fda76fec1fbf310adcbafb7f0d96af13b5511bf8ea954c3aa8eec185734", "x86_64-apple-darwin"),
}
ZIP_NAME = {"windows-x64": "Windows-x64", "macos-arm64": "macOS-arm64", "macos-x64": "macOS-x64"}


def version() -> str:
    return re.search(r'VERSION = "([^"]+)"', (ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")).group(1)


def copytree(src: Path, dst: Path, skip=()) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", *skip), dirs_exist_ok=True)


def write_text(src: Path, dst: Path, eol: str, bom: bool = False) -> None:
    text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
    dst.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.replace("\n", eol).encode("utf-8"))


def uv() -> str:
    return os.environ.get("AICLEAR_UV") or shutil.which("uv") or sys.exit("uv not found; set AICLEAR_UV")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# runtimes

def windows_runtime(rt: Path) -> Path:
    rt.mkdir()
    for item in WIN_PYTHON.iterdir():
        if item.name in {"tcl", "include", "libs", "Scripts"}:
            continue
        if item.name == "Lib":
            copytree(item, rt / "Lib", skip=LIB_SKIP)
        elif item.is_dir():
            copytree(item, rt / item.name)
        else:
            shutil.copy2(item, rt / item.name)
    for dll in VC_DLLS:
        shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / dll, rt / dll)
    site = rt / "Lib" / "site-packages"
    site.mkdir()
    return site


def mac_runtime(rt: Path, target: str) -> Path:
    name, digest, _ = MAC_PYTHON[target]
    tar_path = BUILD / "mac-runtime" / name
    if not tar_path.exists():
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name}")
        urllib.request.urlretrieve(PBS + name.replace("+", "%2B"), tar_path)
    if sha256(tar_path) != digest:
        sys.exit(f"checksum mismatch for {name}")
    skip_dirs = ("python/share/", "python/include/", "python/lib/pkgconfig/", "python/lib/tcl", "python/lib/tk",
                 "python/lib/itcl", "python/lib/thread")
    skip_lib = tuple(f"python/lib/python3.12/{d}/" for d in LIB_SKIP)
    with tarfile.open(tar_path) as t:
        for m in t:
            n = m.name
            if not n.startswith("python/") or m.issym() or m.islnk():
                continue  # symlinks are only convenience aliases (python3 -> python3.12)
            if n.startswith(skip_dirs) or n.startswith(skip_lib) or "_tkinter" in n:
                continue
            if n.startswith("python/bin/") and n != "python/bin/python3.12":
                continue
            dest = rt / n[len("python/"):]
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    site = rt / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


# ---------------------------------------------------------------------------

def build(target: str) -> Path:
    ver = version()
    mac = target.startswith("macos")
    pkg = BUILD / target / "AIClear"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    print(f"[{target}] runtime ...")
    site = mac_runtime(pkg / "runtime", target) if mac else windows_runtime(pkg / "runtime")

    print(f"[{target}] dependencies ...")
    env = dict(os.environ)
    if mac:
        env["MACOSX_DEPLOYMENT_TARGET"] = MAC_MIN  # only accept wheels tagged for <= macOS 13.4
        platform, req = MAC_PYTHON[target][2], "requirements-runtime-macos.txt"
    else:
        platform, req = "x86_64-pc-windows-msvc", "requirements-runtime.txt"
    subprocess.run([uv(), "pip", "install", "--quiet", "--target", str(site), "--python-version", "3.12",
                    "--python-platform", platform, "--only-binary", ":all:",
                    "-r", str(ROOT / "backend" / req), "--index-url", INDEX], check=True, env=env)

    if mac:
        # console-script launchers that uv generated for the build host (Windows .exe) - unused
        shutil.rmtree(site / "bin", ignore_errors=True)

    print(f"[{target}] app, web, models ...")
    copytree(ROOT / "backend" / "app", pkg / "app_src" / "app")
    copytree(ROOT / "frontend" / "dist", pkg / "web")
    (pkg / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy2(ROOT / "models" / f, pkg / "models" / f)
    for f in ("launcher.py", "selftest.py"):
        shutil.copy2(ROOT / "packaging" / f, pkg / f)
    if mac:
        for f in ("Start AIClear.command", "Run Self-Test.command", "ReadMe.txt"):
            write_text(ROOT / "packaging" / "mac" / f, pkg / f, "\n")
        write_text(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt", pkg / "THIRD_PARTY_NOTICES.txt", "\n")
    else:
        # ASCII file names: older Explorer / unzip tools ignore the zip UTF-8 flag
        write_text(ROOT / "packaging" / "Start AIClear.bat", pkg / "Start AIClear.bat", "\r\n")
        write_text(ROOT / "packaging" / "ReadMe.txt", pkg / "ReadMe.txt", "\r\n", bom=True)
        write_text(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt", pkg / "THIRD_PARTY_NOTICES.txt", "\r\n")

    print(f"[{target}] byte-compiling ...")
    compileall.compile_dir(str(pkg), quiet=1, workers=0, legacy=False)

    if mac:
        print(f"[{target}] checking minimum macOS of all binaries ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "macho_minos.py"), str(pkg), "--max", MAC_MIN], check=True)

    size = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    zpath = BUILD / f"AIClear-v{ver}-{ZIP_NAME[target]}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(pkg.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(pkg)
            zi = zipfile.ZipInfo.from_file(f, str(Path("AIClear") / rel))
            zi.compress_type = zipfile.ZIP_DEFLATED
            if mac:
                # Unix permissions so Finder / Archive Utility keep scripts and Python executable
                exe = f.suffix in (".command", ".so", ".dylib") or rel.as_posix() == "runtime/bin/python3.12"
                zi.create_system = 3
                zi.external_attr = ((0o100755 if exe else 0o100644) & 0xFFFF) << 16
            with open(f, "rb") as fh:
                z.writestr(zi, fh.read(), compresslevel=6)
    print(f"[{target}] folder {size / 1e6:.0f} MB -> {zpath.name} ({zpath.stat().st_size / 1e6:.0f} MB)")
    return zpath


def main() -> None:
    for f in MODEL_FILES:
        if not (ROOT / "models" / f).exists():
            sys.exit(f"missing models/{f} - run scripts/download_models.py and scripts/export_onnx.py first")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        sys.exit("frontend not built - run npm run build in frontend/")
    targets = [a for a in sys.argv[1:] if not a.startswith("-")] or list(ZIP_NAME)
    for t in targets:
        if t not in ZIP_NAME:
            sys.exit(f"unknown target {t}; choose from {list(ZIP_NAME)}")
        build(t)


if __name__ == "__main__":
    main()
