"""Export the tier models to ONNX (fixed tile size) and verify them against PyTorch.

The packaged app runs ONNX Runtime (DirectML on any DX12 GPU, otherwise CPU), so the
release needs no PyTorch/CUDA. Run once on the dev machine:

    backend/.venv/Scripts/python scripts/export_onnx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import config  # noqa: E402
from app.imaging.sr import MODELS, ONNX_TILE  # noqa: E402

EXPORT = ["SwinIR-M_x4_PSNR", "SwinIR-M_x4_GAN"]


def main() -> None:
    from spandrel import ModelLoader
    import onnxruntime as ort

    names = [a for a in sys.argv[1:] if not a.startswith("--")] or EXPORT
    for name in names:
        meta = MODELS[name]
        desc = ModelLoader().load_from_file(config.MODELS_DIR / meta["file"])
        net = desc.model.eval().float()
        out_path = config.MODELS_DIR / meta["onnx"]
        if not out_path.exists() or "--force" in sys.argv:
            x = torch.rand(1, 3, ONNX_TILE, ONNX_TILE)
            t0 = time.time()
            torch.onnx.export(net, (x,), str(out_path), input_names=["input"], output_names=["output"],
                              opset_version=17, dynamo=False)
            print(f"{name}: exported {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")

        rng = np.random.default_rng(0)
        sample = torch.from_numpy(rng.random((1, 3, ONNX_TILE, ONNX_TILE), dtype=np.float32))
        with torch.inference_mode():
            ref = net(sample).numpy()
        for prov in ("DmlExecutionProvider", "CPUExecutionProvider"):
            so = ort.SessionOptions()
            so.enable_mem_pattern = False  # required by the DirectML EP
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(str(out_path), so, providers=[prov])
            sess.run(None, {"input": sample.numpy()})  # warm-up
            t0 = time.time()
            got = sess.run(None, {"input": sample.numpy()})[0]
            dt = time.time() - t0
            err = np.abs(got - ref).max()
            psnr = 10 * np.log10(1 / max(float(np.mean((got - ref) ** 2)), 1e-12))
            print(f"  {prov:22s} {dt:.2f}s/tile  max|diff| {err:.4f}  PSNR vs torch {psnr:.1f} dB")


if __name__ == "__main__":
    main()
