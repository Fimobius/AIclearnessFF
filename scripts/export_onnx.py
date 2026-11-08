"""Export the tier models to ONNX (fixed tile size) and verify them against PyTorch.

The packaged apps run ONNX Runtime (DirectML on Windows, CoreML on macOS, otherwise CPU),
so the releases need no PyTorch. Run once on the dev machine:

    backend/.venv/Scripts/python scripts/export_onnx.py [--force] [model names]

Before export, SwinIR is rewritten into a mathematically identical, accelerator-friendly
form for the fixed 256x256 tile (batch 1). Checked with
`python -m onnxruntime.tools.check_onnx_model_mobile_usability`, the stock export splits into
111 CoreML partitions; the causes and fixes are:
  - q, k, v = qkv[0], qkv[1], qkv[2] exports as scalar Gather (unsupported) -> torch.split
  - window partition / reverse use 6-D reshapes (CoreML max rank 5)       -> 5-D (batch 1)
  - relative-position bias lookup and shifted-window masks are computed at run time
    -> pre-computed constants for the fixed tile size
  - shape arithmetic around Reshape/Resize is computed at run time, so CoreML sees dynamic
    shapes -> folded into constants by ONNX Runtime's basic optimisation (finalize_graph)
The rewritten model is compared with the original PyTorch model before saving.
"""
from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import config  # noqa: E402
from app.imaging.sr import MODELS, ONNX_TILE  # noqa: E402

EXPORT = ["SwinIR-M_x4_PSNR", "SwinIR-M_x4_GAN"]


def _window_partition_b1(x, window_size):
    B, H, W, C = x.shape
    assert B == 1, "export assumes batch size 1"
    x = x.reshape(H // window_size, window_size, W // window_size, window_size, C).permute(0, 2, 1, 3, 4)
    return x.reshape(-1, window_size, window_size, C)


def _window_reverse_b1(windows, window_size, H, W):
    C = windows.shape[-1]
    x = windows.reshape(H // window_size, W // window_size, window_size, window_size, C).permute(0, 2, 1, 3, 4)
    return x.reshape(1, H, W, C)


def _attention_forward(self, x, mask=None):
    """WindowAttention.forward without scalar Gather and with the pre-computed bias."""
    B_, N, C = x.shape
    nh = self.num_heads
    qkv = self.qkv(x).reshape(B_, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
    q, k, v = (t.reshape(B_, nh, N, C // nh) for t in torch.split(qkv, 1, dim=0))
    attn = (q * self.scale) @ k.transpose(-2, -1) + self._rel_pos_bias
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, nh, N, N)
    attn = self.softmax(attn)
    return self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C))


def prepare_for_export(net: torch.nn.Module, tile: int) -> torch.nn.Module:
    mod = importlib.import_module(type(net).__module__)
    mod.window_partition = _window_partition_b1  # module globals used by SwinTransformerBlock
    mod.window_reverse = _window_reverse_b1
    for m in net.modules():
        if isinstance(m, mod.WindowAttention):
            n = m.window_size[0] * m.window_size[1]
            bias = m.relative_position_bias_table[m.relative_position_index.view(-1)].view(n, n, -1)
            m.register_buffer("_rel_pos_bias", bias.permute(2, 0, 1).contiguous().unsqueeze(0).detach().clone())
            m.forward = types.MethodType(_attention_forward, m)
        elif isinstance(m, mod.SwinTransformerBlock):
            # the feature map equals the tile (256 is a multiple of the window size 8)
            m.input_resolution = (tile, tile)
            m.attn_mask = m.calculate_mask((tile, tile)) if m.shift_size > 0 else None
    return net


def finalize_graph(path: Path) -> None:
    """Fold all constant shape arithmetic with ONNX Runtime's *basic* (provider-independent)
    graph optimisations and save the result as the shipped model. Basic level = constant
    folding and redundancy elimination only, so the file stays portable across EPs/OSes."""
    import onnxruntime as ort

    tmp = path.with_suffix(".opt.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(tmp)
    ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    tmp.replace(path)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main() -> None:
    from spandrel import ModelLoader
    import onnxruntime as ort

    names = [a for a in sys.argv[1:] if not a.startswith("--")] or EXPORT
    rng = np.random.default_rng(0)
    sample = torch.from_numpy(rng.random((1, 3, ONNX_TILE, ONNX_TILE), dtype=np.float32))
    for name in names:
        meta = MODELS[name]
        path = config.MODELS_DIR / meta["file"]
        with torch.no_grad():  # not inference_mode: SwinIR caches tensors on itself in forward
            ref = ModelLoader().load_from_file(path).model.eval().float()(sample).numpy()  # original model
        out_path = config.MODELS_DIR / meta["onnx"]
        if not out_path.exists() or "--force" in sys.argv:
            net = prepare_for_export(ModelLoader().load_from_file(path).model.eval().float(), ONNX_TILE)
            with torch.no_grad():  # not inference_mode: SwinIR caches tensors on itself in forward
                patched = net(sample).numpy()
            print(f"{name}: rewritten model vs original (PyTorch): PSNR {_psnr(patched, ref):.1f} dB")
            assert _psnr(patched, ref) > 80, "rewrite changed the model's output"
            t0 = time.time()
            with torch.no_grad():  # tracing with autograd on crashes (access violation) for the rewritten model
                torch.onnx.export(net, (sample,), str(out_path), input_names=["input"], output_names=["output"],
                                  opset_version=17, dynamo=False, do_constant_folding=True)
            finalize_graph(out_path)
            print(f"{name}: exported {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
        for prov in ("DmlExecutionProvider", "CPUExecutionProvider"):
            if prov not in ort.get_available_providers():
                continue
            so = ort.SessionOptions()
            so.enable_mem_pattern = False  # required by the DirectML EP
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(str(out_path), so, providers=[prov])
            sess.run(None, {"input": sample.numpy()})  # warm-up
            t0 = time.time()
            got = sess.run(None, {"input": sample.numpy()})[0]
            print(f"  {prov:22s} {time.time() - t0:.2f}s/tile  PSNR vs original PyTorch {_psnr(got, ref):.1f} dB")


if __name__ == "__main__":
    main()
