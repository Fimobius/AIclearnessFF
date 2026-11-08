import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Isolated data dir for every test session; must be set before importing app modules
os.environ.setdefault("AICLEAR_DATA_DIR", tempfile.mkdtemp(prefix="aiclear-test-"))
os.environ.setdefault("AICLEAR_RETENTION_DAYS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_image(w=96, h=64, mode="RGB", fmt="JPEG", seed=0, **save_kw) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = np.linspace(0, 255, w)[None, :]
    arr[..., 1] = np.linspace(0, 255, h)[:, None]
    arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = (200, 30, 40)
    arr = np.clip(arr + rng.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if mode != "RGB":
        im = im.convert(mode)
    buf = io.BytesIO()
    im.save(buf, fmt, **save_kw)
    return buf.getvalue()


@pytest.fixture(scope="session")
def engine():
    from app.imaging.sr import get_engine
    return get_engine()
