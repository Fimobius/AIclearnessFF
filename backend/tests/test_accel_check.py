"""The ONNX engine must only keep a hardware accelerator that is both correct and faster."""
import json
import time

import numpy as np
import pytest

from app import config
from app.imaging import sr

ort = pytest.importorskip("onnxruntime")


class FakeSession:
    def __init__(self, fn, delay=0.0, fail=False):
        self.fn, self.delay, self.fail = fn, delay, fail

    def run(self, _outputs, feeds):
        if self.fail:
            raise RuntimeError("driver crashed")
        time.sleep(self.delay)
        x = feeds["input"]
        return [self.fn(np.repeat(np.repeat(x, 4, axis=2), 4, axis=3))]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    eng = sr.OnnxEngine(force_cpu=True)
    eng.accel_name, eng._accel_providers, eng.use_accel = "FakeGPU", ["fake"], True
    eng._apply_device()
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    return eng, str(model)


def _with_cpu(eng, monkeypatch, delay):
    monkeypatch.setattr(eng, "_session", lambda path, accel: FakeSession(lambda y: y, delay=delay))


def test_correct_and_faster_accelerator_is_kept(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.05)
    assert eng._verify_accel(path, FakeSession(lambda y: y))
    saved = json.loads((config.DATA_DIR / "accel_check.json").read_text())
    assert list(saved.values())[0]["ok"] is True


def test_wrong_output_is_rejected(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.05)
    assert not eng._verify_accel(path, FakeSession(lambda y: y * 0.5))
    assert "differs" in eng.accel_checks["m.onnx"]["reason"]


def test_slower_accelerator_is_rejected(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.0)
    assert not eng._verify_accel(path, FakeSession(lambda y: y, delay=0.05))
    assert "not faster" in eng.accel_checks["m.onnx"]["reason"]


def test_crashing_accelerator_is_rejected(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.0)
    assert not eng._verify_accel(path, FakeSession(lambda y: y, fail=True))
    assert "error" in eng.accel_checks["m.onnx"]["reason"]


def test_verdict_is_cached(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.05)
    assert eng._verify_accel(path, FakeSession(lambda y: y))
    # second call must not run anything: a crashing session would otherwise flip the verdict
    assert eng._verify_accel(path, FakeSession(lambda y: y, fail=True))


def test_rejection_switches_engine_to_cpu_settings(engine, monkeypatch):
    eng, path = engine
    _with_cpu(eng, monkeypatch, delay=0.0)
    monkeypatch.setitem(sr.MODELS, "Fake", {"scale": 4, "onnx": "m.onnx", "file": "m.pth"})
    monkeypatch.setattr(config, "MODELS_DIR", config.DATA_DIR)
    eng._session = lambda p, accel: FakeSession(lambda y: y * (0.5 if accel else 1.0))
    eng._load(sr.ModelSpec("Fake"))
    assert eng.use_accel is False and eng.supersample == 1.0
    assert eng.device_info()["device"] == "cpu"
