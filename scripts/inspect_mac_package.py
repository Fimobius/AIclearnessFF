"""Static checks of a macOS release zip, runnable on any OS.

- every Mach-O binary is for the expected architecture and needs <= macOS 13.4
- no Windows binaries (.dll/.pyd/.exe) slipped in
- start scripts / Python are executable and use LF line endings
- the pinned dependency versions were installed

Usage: python scripts/inspect_mac_package.py build/AIClear-v1.1.0-macOS-arm64.zip arm64
"""
from __future__ import annotations

import re
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from macho_minos import macho_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX = (13, 4)


def main() -> None:
    zpath, arch = Path(sys.argv[1]), sys.argv[2]
    z = zipfile.ZipFile(zpath)
    problems: list[str] = []
    archs: Counter = Counter()
    for zi in z.infolist():
        n = zi.filename
        if n.lower().endswith((".dll", ".pyd", ".exe")):
            problems.append(f"Windows binary: {n}")
        data = z.read(n) if n.endswith((".so", ".dylib")) or n.endswith("/python3.12") else None
        if data:
            slices = dict(macho_info(data))  # universal2 binaries contain both archs
            archs["universal2" if len(slices) > 1 else next(iter(slices), "?")] += 1
            if arch not in slices:
                problems.append(f"no {arch} code: {n} ({list(slices)})")
            minos = slices.get(arch)
            if minos and tuple(map(int, minos.split("."))) > MAX:
                problems.append(f"needs macOS {minos}: {n}")
    for script in ("AIClear/Start AIClear.command", "AIClear/Run Self-Test.command", "AIClear/runtime/bin/python3.12"):
        zi = z.getinfo(script)
        mode = zi.external_attr >> 16
        if zi.create_system != 3 or not mode & stat.S_IXUSR:
            problems.append(f"not executable: {script} (mode {oct(mode)})")
        if script.endswith(".command") and b"\r\n" in z.read(script):
            problems.append(f"CRLF line endings: {script}")
    pins = dict(re.findall(r"^([\w-]+)==([\w.]+)", (ROOT / "backend" / "requirements-runtime-macos.txt").read_text(), re.M))
    installed = {}
    for n in z.namelist():
        m = re.match(r"AIClear/runtime/lib/python3\.12/site-packages/([\w.]+)-([\w.]+)\.dist-info/METADATA$", n)
        if m:
            installed[m.group(1).lower().replace("_", "-")] = m.group(2)
    for pkg, ver in pins.items():
        if installed.get(pkg.lower()) != ver:
            problems.append(f"{pkg}: expected {ver}, found {installed.get(pkg.lower())}")
    print(f"{zpath.name}: {len(z.infolist())} files, Mach-O binaries by arch {dict(archs)}")
    print("pinned versions:", {k: installed.get(k.lower()) for k in pins})
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems[:30]))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
