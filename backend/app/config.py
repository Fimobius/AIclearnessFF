"""Paths and limits. Everything can be overridden with AICLEAR_* environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    raw = os.environ.get(f"AICLEAR_{name}")
    if raw is None:
        return default
    return type(default)(raw) if not isinstance(default, Path) else Path(raw)


DATA_DIR: Path = _env("DATA_DIR", ROOT / "data")
MODELS_DIR: Path = _env("MODELS_DIR", ROOT / "models")
FRONTEND_DIST: Path = _env("FRONTEND_DIST", ROOT / "frontend" / "dist")

# Upload limits
MAX_FILE_MB: int = _env("MAX_FILE_MB", 30)
MAX_FILES_PER_JOB: int = _env("MAX_FILES_PER_JOB", 500)
MAX_JOB_MB: int = _env("MAX_JOB_MB", 2048)
MAX_INPUT_PIXELS: int = _env("MAX_INPUT_PIXELS", 50_000_000)

# Output size limits
MIN_SIDE: int = 16
MAX_SIDE: int = _env("MAX_SIDE", 10000)
MAX_OUTPUT_PIXELS: int = _env("MAX_OUTPUT_PIXELS", 40_000_000)

# Inference
TILE_SIZE: int = _env("TILE_SIZE", 512)
TILE_OVERLAP: int = _env("TILE_OVERLAP", 32)
DEVICE: str = _env("DEVICE", "auto")  # auto | cpu  (cpu forces CPU on either backend)
BACKEND: str = _env("BACKEND", "auto")  # auto | torch | onnx  (auto: torch if CUDA is present)

# Whether "stretch" resize mode is offered (PRD: optional, can be switched off)
ALLOW_STRETCH: bool = _env("ALLOW_STRETCH", "1") not in ("0", "false", "False")

# Optional API key. When set, every /api request must send header X-API-Key.
API_KEY: str = _env("API_KEY", "")

# Finished jobs older than this are purged at startup (0 = keep forever)
RETENTION_DAYS: int = _env("RETENTION_DAYS", 30)

# Fidelity guard thresholds (calibrated on the golden set, see eval/README.md)
QA_MAX_DELTA_E: float = _env("QA_MAX_DELTA_E", 3.0)
QA_MIN_SSIM: float = _env("QA_MIN_SSIM", 0.85)
