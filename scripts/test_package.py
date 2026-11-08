"""Run the in-package self-test (packaging/selftest.py) of an extracted release package
with the package's own Python and a stripped-down environment.

Usage: backend/.venv/Scripts/python scripts/test_package.py <extracted AIClear dir> [--cpu]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    pkg = Path(sys.argv[1]).resolve()
    env = dict(os.environ)
    if "--cpu" in sys.argv:
        env["AICLEAR_DEVICE"] = "cpu"
    py = pkg / "runtime" / ("python.exe" if os.name == "nt" else "bin/python3.12")
    sys.exit(subprocess.run([str(py), "-X", "utf8", str(pkg / "selftest.py")], cwd=pkg, env=env).returncode)


if __name__ == "__main__":
    main()
