"""Report the minimum macOS version (LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX) of every
Mach-O binary in a directory or tar archive - lets us check macOS compatibility of a Mac
package from any OS.

Usage: python scripts/macho_minos.py <dir | .tar.gz> [--max 13.0]
Exits non-zero if --max is given and any binary requires a newer macOS.
"""
from __future__ import annotations

import io
import struct
import sys
import tarfile
from collections import Counter
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
CPU = {0x01000007: "x86_64", 0x0100000C: "arm64"}


def _ver(v: int) -> str:
    return f"{v >> 16}.{(v >> 8) & 0xFF}"


def _thin(data: bytes, off: int = 0) -> tuple[str, str | None] | None:
    if len(data) < off + 32 or struct.unpack_from("<I", data, off)[0] != MH_MAGIC_64:
        return None
    cputype, _, _, ncmds, _, _, _ = struct.unpack_from("<iiIIIII", data, off + 4)
    p = off + 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, p)
        if cmd == LC_BUILD_VERSION:
            return CPU.get(cputype & 0xFFFFFFFF, hex(cputype)), _ver(struct.unpack_from("<I", data, p + 12)[0])
        if cmd == LC_VERSION_MIN_MACOSX:
            return CPU.get(cputype & 0xFFFFFFFF, hex(cputype)), _ver(struct.unpack_from("<I", data, p + 8)[0])
        p += size
    return CPU.get(cputype & 0xFFFFFFFF, hex(cputype)), None


def macho_info(data: bytes) -> list[tuple[str, str | None]]:
    if len(data) >= 8 and struct.unpack_from(">I", data)[0] == FAT_MAGIC:
        n = struct.unpack_from(">I", data, 4)[0]
        out = []
        for i in range(n):
            _, _, offset, _, _ = struct.unpack_from(">iiIII", data, 8 + i * 20)
            r = _thin(data, offset)
            if r:
                out.append(r)
        return out
    r = _thin(data)
    return [r] if r else []


def iter_files(src: Path):
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file() and not f.is_symlink():
                with open(f, "rb") as fh:
                    head = fh.read(4)
                    if head in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
                        yield str(f.relative_to(src)), head + fh.read()
    else:
        with tarfile.open(src) as t:
            for m in t:
                if m.isfile():
                    data = t.extractfile(m).read()
                    if data[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
                        yield m.name, data


def main() -> None:
    src = Path(sys.argv[1])
    limit = None
    if "--max" in sys.argv:
        limit = tuple(int(x) for x in sys.argv[sys.argv.index("--max") + 1].split("."))
    counts: Counter = Counter()
    worst: list[tuple[tuple[int, ...], str, str, str]] = []
    for name, data in iter_files(src):
        for arch, minos in macho_info(data):
            counts[(arch, minos)] += 1
            if minos:
                worst.append((tuple(int(x) for x in minos.split(".")), minos, arch, name))
    print("binaries by (arch, min macOS):", dict(counts))
    worst.sort(reverse=True)
    for _, minos, arch, name in worst[:5]:
        print(f"  newest requirement: macOS {minos} {arch} {name}")
    if limit and worst and worst[0][0] > limit:
        too_new = [w for w in worst if w[0] > limit]
        print(f"FAIL: {len(too_new)} binaries need macOS newer than {'.'.join(map(str, limit))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
