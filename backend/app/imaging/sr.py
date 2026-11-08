"""Super-resolution engine: model registry and tiled inference on two backends.

- "torch": PyTorch + spandrel, CUDA if available (development machine; fp16/bf16).
- "onnx":  ONNX Runtime with DirectML on Windows (any DX12 GPU: NVIDIA / AMD / Intel) or
           CoreML on macOS (Apple GPU / Neural Engine), with automatic CPU fallback. Used by
           the packaged apps - no PyTorch needed.
Selected by AICLEAR_BACKEND = auto | torch | onnx (auto: torch+CUDA if present, else onnx).
"""
from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .. import config

log = logging.getLogger("aiclear.sr")

# Only weights whose licence allows commercial use are registered here.
MODELS: dict[str, dict] = {
    "RealESRGAN_x4plus": {
        "file": "RealESRGAN_x4plus.pth", "scale": 4, "license": "BSD-3-Clause",
        "source": "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0",
        "desc": "Real-ESRGAN x4（GAN，细节强，适合重度模糊）",
    },
    "RealESRNet_x4plus": {
        "file": "RealESRNet_x4plus.pth", "scale": 4, "license": "BSD-3-Clause",
        "source": "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1",
        "desc": "Real-ESRNet x4（无 GAN，保真优先、较柔和）",
    },
    "RealESRGAN_x2plus": {
        "file": "RealESRGAN_x2plus.pth", "scale": 2, "license": "BSD-3-Clause",
        "source": "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1",
        "desc": "Real-ESRGAN x2",
    },
    "realesr-general-x4v3": {
        "file": "realesr-general-x4v3.pth", "scale": 4, "license": "BSD-3-Clause",
        "source": "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0",
        "desc": "Real-ESRGAN general v3（轻量，降噪强度可调）",
        "dni_partner": "realesr-general-wdn-x4v3.pth",
    },
    "SwinIR-M_x4_GAN": {
        "file": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", "scale": 4, "license": "Apache-2.0",
        "source": "https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0",
        "desc": "SwinIR-M 真实场景 x4（GAN）", "onnx": "SwinIR-M_x4_GAN_t256.onnx",
    },
    "SwinIR-M_x4_PSNR": {
        "file": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth", "scale": 4, "license": "Apache-2.0",
        "source": "https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0",
        "desc": "SwinIR-M 真实场景 x4（PSNR，保真优先）", "onnx": "SwinIR-M_x4_PSNR_t256.onnx",
    },
}

# ONNX exports use a fixed input tile (SwinIR's attention masks depend on the input size)
ONNX_TILE = 256


@dataclass(frozen=True)
class ModelSpec:
    """A concrete model configuration. `denoise` only applies to DNI-capable models."""
    name: str
    denoise: float | None = None

    @property
    def key(self) -> str:
        return self.name if self.denoise is None else f"{self.name}@dn{self.denoise:.2f}"


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def available_models() -> dict[str, dict]:
    onnx = isinstance(get_engine(), OnnxEngine)
    out = {}
    for name, m in MODELS.items():
        if onnx:
            ok = "onnx" in m and (config.MODELS_DIR / m["onnx"]).exists()
        else:
            partner = m.get("dni_partner")
            ok = (config.MODELS_DIR / m["file"]).exists() and (partner is None or (config.MODELS_DIR / partner).exists())
        out[name] = {**m, "available": ok}
    return out


# ---------------------------------------------------------------------------
# shared tiling

def _split(n: int, tile: int, ov: int) -> tuple[list[int], int]:
    """Evenly sized tiles: the fewest tiles <= `tile` covering n with `ov` overlap.
    (Fixed-size tiles waste up to ~4x compute when n is slightly above `tile`.)"""
    if n <= tile:
        return [0], n
    k = math.ceil((n - ov) / (tile - ov))
    t = math.ceil((n + (k - 1) * ov) / k)
    return [min(i * (t - ov), n - t) for i in range(k)], t


def _feather(h: int, w: int, ov: int, top: bool, bottom: bool, left: bool, right: bool) -> np.ndarray:
    def ramp(n: int, start: bool, end: bool) -> np.ndarray:
        r = np.ones(n, np.float32)
        k = min(ov, n // 2)
        if k > 0:
            lin = (np.arange(k, dtype=np.float32) + 0.5) / k
            if start:
                r[:k] = lin
            if end:
                r[n - k:] = lin[::-1]
        return r
    return ramp(h, top, bottom)[:, None] * ramp(w, left, right)[None, :]


def run_tiled(rgb: np.ndarray, scale: int, tile: int, ov: int,
              infer: Callable[[np.ndarray], np.ndarray],
              progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """rgb uint8 (H, W, 3) -> uint8 (H*s, W*s, 3). `infer` maps a float32 (1,3,h,w) patch in
    [0,1] to (1,3,h*s,w*s); overlapping tiles are blended with linear feathering."""
    h, w = rgb.shape[:2]
    ov = min(ov, tile // 4)
    img = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    (ys, th), (xs, tw) = _split(h, tile, ov), _split(w, tile, ov)
    out = np.zeros((3, h * scale, w * scale), np.float32)
    weight = np.zeros((h * scale, w * scale), np.float32)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            res = infer(np.ascontiguousarray(img[:, :, y:y + th, x:x + tw]))[0, :, :th * scale, :tw * scale]
            mask = _feather(th * scale, tw * scale, ov * scale, y > 0, y + th < h, x > 0, x + tw < w)
            out[:, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += np.clip(res, 0, 1) * mask
            weight[y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += mask
            done += 1
            if progress:
                progress(done, total)
    out /= np.maximum(weight, 1e-6)
    return (out.transpose(1, 2, 0) * 255 + 0.5).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# engines

class SREngine:
    """Common interface. Inference is serialised (one GPU)."""
    backend = "base"
    torch_device = None  # set by the torch engine; enables the GPU colour fix
    supersample = 2.0  # SR output ~ this x the needed size (suppresses AI artefacts)
    default_rate = 0.62  # seconds per SR-output megapixel before anything was measured

    def __init__(self, max_cached: int = 2):
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_cached = max_cached
        self._lock = threading.Lock()

    def scale_of(self, spec: ModelSpec) -> int:
        return MODELS[spec.name]["scale"]

    def get(self, spec: ModelSpec):
        k = spec.key
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        while len(self._cache) >= self._max_cached:
            self._cache.popitem(last=False)
            self._evicted()
        m = self._load(spec)
        self._cache[k] = m
        return m

    def _evicted(self) -> None:
        pass

    def _load(self, spec: ModelSpec):
        raise NotImplementedError

    def device_info(self) -> dict:
        raise NotImplementedError

    def upscale(self, rgb: np.ndarray, spec: ModelSpec,
                progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        raise NotImplementedError


class TorchEngine(SREngine):
    backend = "torch"

    def __init__(self, device: str = "cuda"):
        super().__init__()
        import torch
        self.torch_device = torch.device(device)
        if self.torch_device.type == "cpu":
            self.supersample, self.default_rate = 1.0, 8.0

    def device_info(self) -> dict:
        import torch
        if self.torch_device.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            return {"device": "cuda", "backend": "PyTorch CUDA", "name": p.name,
                    "vram_gb": round(p.total_memory / 1024**3, 1)}
        return {"device": "cpu", "backend": "PyTorch CPU", "name": "CPU（无可用 GPU，处理会很慢）"}

    def _evicted(self) -> None:
        import torch
        if self.torch_device.type == "cuda":
            torch.cuda.empty_cache()

    def _load(self, spec: ModelSpec):
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        if spec.name not in MODELS:
            raise KeyError(f"未知模型 {spec.name}")
        meta = MODELS[spec.name]
        path = config.MODELS_DIR / meta["file"]
        loader = ModelLoader()
        if spec.denoise is not None and meta.get("dni_partner"):
            # Deep network interpolation (Real-ESRGAN's --denoise_strength):
            # weight dn on the strong-denoise net, 1-dn on the weak-denoise (wdn) net.
            sd_a = torch.load(path, map_location="cpu", weights_only=True)
            sd_b = torch.load(config.MODELS_DIR / meta["dni_partner"], map_location="cpu", weights_only=True)
            key = "params_ema" if "params_ema" in sd_a else ("params" if "params" in sd_a else None)
            a = sd_a[key] if key else sd_a
            b = sd_b[key] if key else sd_b
            dn = float(np.clip(spec.denoise, 0, 1))
            desc = loader.load_from_state_dict({k: dn * a[k] + (1 - dn) * b[k] for k in a})
        else:
            desc = loader.load_from_file(path)
        if not isinstance(desc, ImageModelDescriptor):
            raise TypeError(f"{spec.name} 不是图像超分模型")
        desc.to(self.torch_device).eval()
        half = self.torch_device.type == "cuda" and desc.supports_half
        if half:
            desc.half()
        return desc, half

    def upscale(self, rgb, spec, progress=None):
        import contextlib
        import torch
        with self._lock:
            desc, half = self.get(spec)
            req = desc.size_requirements
            dtype = torch.float16 if half else torch.float32
            cuda = self.torch_device.type == "cuda"
            # Models without fp16 support (SwinIR) run under bf16 autocast: ~1.4x faster,
            # output within 44-54 dB PSNR of fp32 (measured on the golden set) - invisible.
            use_bf16 = not half and cuda and torch.cuda.is_bf16_supported()

            @torch.inference_mode()
            def infer(patch: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(patch)
                h, w = t.shape[2:]
                ph = -(-max(h, req.minimum) // req.multiple_of) * req.multiple_of
                pw = -(-max(w, req.minimum) // req.multiple_of) * req.multiple_of
                if (ph, pw) != (h, w):
                    t = torch.nn.functional.pad(t, (0, pw - w, 0, ph - h), mode="replicate")
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                with ctx:
                    return desc(t.to(self.torch_device, dtype)).float().cpu().numpy()

            tile = config.TILE_SIZE
            while True:
                try:
                    return run_tiled(rgb, desc.scale, tile, config.TILE_OVERLAP, infer, progress)
                except torch.OutOfMemoryError:
                    if cuda:
                        torch.cuda.empty_cache()
                    if tile <= 64:
                        raise
                    tile //= 2


class OnnxEngine(SREngine):
    """ONNX Runtime with a hardware-accelerated execution provider where available:
    DirectML on Windows (any DX12 GPU; prefers the high-performance adapter on laptops with
    two GPUs), CoreML on macOS (Apple GPU / Neural Engine). Falls back to CPU.

    The accelerated path is only trusted after a one-time check per model: one tile is run on
    both the accelerator and the CPU; it is kept only if the output matches (PSNR >= 40 dB)
    and it is actually faster. CoreML in particular may not support every SwinIR op (6-D
    window reshapes) and then partitions the graph, which can be slower than plain CPU.
    The verdict is cached in DATA_DIR/accel_check.json.
    """
    backend = "onnx"
    MIN_PSNR = 40.0

    def __init__(self, force_cpu: bool = False):
        super().__init__()
        import onnxruntime as ort
        self._ort = ort
        self.accel_name, self._accel_providers = (None, []) if force_cpu else self._detect_accel()
        self.use_accel = bool(self._accel_providers)
        self.accel_checks: dict[str, dict] = {}
        self._apply_device()

    def _detect_accel(self) -> tuple[str | None, list]:
        avail = self._ort.get_available_providers()
        if sys.platform == "win32" and "DmlExecutionProvider" in avail:
            return "DirectML", [("DmlExecutionProvider", {"performance_preference": "high_performance"})]
        if sys.platform == "darwin" and "CoreMLExecutionProvider" in avail:
            cache = config.DATA_DIR / "coreml_cache"
            cache.mkdir(parents=True, exist_ok=True)
            return "CoreML", [("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1", "ModelCacheDirectory": str(cache)})]
        return None, []

    def _apply_device(self) -> None:
        if self.use_accel:
            self.supersample, self.default_rate = 2.0, 0.7
        else:
            # CPU is ~13x slower: drop supersampling (4x less work) to keep it usable
            self.supersample, self.default_rate = 1.0, 8.0

    def _fallback(self, why: str) -> None:
        log.warning("%s disabled (%s); using CPU", self.accel_name, why)
        self.use_accel = False
        self._apply_device()

    def device_info(self) -> dict:
        if self.use_accel:
            info = {"device": "gpu", "backend": self.accel_name, "name": f"GPU 加速（{self.accel_name}）"}
        else:
            info = {"device": "cpu", "backend": "CPU", "name": "CPU（未检测到可用的加速硬件，处理较慢）"}
        if self.accel_checks:
            info["accel_checks"] = self.accel_checks
        return info

    def _session(self, path: str, accel: bool):
        so = self._ort.SessionOptions()
        so.enable_mem_pattern = False  # required by the DirectML EP
        so.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        providers = [*self._accel_providers, "CPUExecutionProvider"] if accel else ["CPUExecutionProvider"]
        return self._ort.InferenceSession(path, so, providers=providers)

    # -- one-time correctness / speed check of the accelerated path --------
    @staticmethod
    def _check_file() -> Path:
        return config.DATA_DIR / "accel_check.json"

    def _check_key(self, path: str) -> str:
        st = Path(path).stat()
        return f"{self.accel_name}|ort{self._ort.__version__}|{Path(path).name}|{st.st_size}|{int(st.st_mtime)}"

    def _verify_accel(self, path: str, accel_sess) -> bool:
        key = self._check_key(path)
        try:
            cache = json.loads(self._check_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            self.accel_checks[Path(path).name] = cache[key]
            return cache[key]["ok"]
        x = np.random.default_rng(0).random((1, 3, ONNX_TILE, ONNX_TILE), dtype=np.float32)
        try:
            accel_sess.run(None, {"input": x})  # warm-up (CoreML compiles on first run)
            t0 = time.perf_counter()
            a = accel_sess.run(None, {"input": x})[0]
            t_accel = time.perf_counter() - t0
        except Exception as e:
            result = {"ok": False, "reason": f"error: {e}"[:300]}
        else:
            cpu = self._session(path, False)
            t0 = time.perf_counter()
            c = cpu.run(None, {"input": x})[0]
            t_cpu = time.perf_counter() - t0
            mse = float(np.mean((np.clip(a, 0, 1) - np.clip(c, 0, 1)) ** 2))
            psnr = 99.0 if mse == 0 else 10 * math.log10(1 / mse)
            ok = psnr >= self.MIN_PSNR and t_accel < 0.9 * t_cpu
            reason = "" if ok else ("output differs from CPU" if psnr < self.MIN_PSNR else "not faster than CPU")
            result = {"ok": ok, "psnr_db": round(psnr, 1), "accel_s": round(t_accel, 3), "cpu_s": round(t_cpu, 3),
                      "reason": reason}
        log.info("accelerator check %s: %s", key, result)
        cache[key] = result
        try:
            self._check_file().parent.mkdir(parents=True, exist_ok=True)
            self._check_file().write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError:
            pass
        self.accel_checks[Path(path).name] = result
        return result["ok"]

    def _load(self, spec: ModelSpec):
        meta = MODELS.get(spec.name)
        if not meta or "onnx" not in meta or spec.denoise is not None:
            raise KeyError(f"模型 {spec.name} 没有 ONNX 版本")
        path = str(config.MODELS_DIR / meta["onnx"])
        if self.use_accel:
            try:
                sess = self._session(path, True)
                if self._verify_accel(path, sess):
                    return sess
                self._fallback(self.accel_checks[Path(path).name]["reason"])
            except Exception as e:  # driver/adapter problems: fall back for good
                self._fallback(str(e))
        return self._session(path, False)

    def upscale(self, rgb, spec, progress=None):
        with self._lock:
            T = ONNX_TILE

            def make_infer(sess):
                def infer(patch: np.ndarray) -> np.ndarray:
                    h, w = patch.shape[2:]
                    if (h, w) != (T, T):
                        patch = np.pad(patch, ((0, 0), (0, 0), (0, T - h), (0, T - w)), mode="edge")
                    return sess.run(None, {"input": patch})[0]
                return infer

            try:
                return run_tiled(rgb, self.scale_of(spec), T, config.TILE_OVERLAP,
                                 make_infer(self.get(spec)), progress)
            except Exception as e:
                if not self.use_accel:
                    raise
                self._fallback(f"inference failed: {e}")
                self._cache.clear()
                return run_tiled(rgb, self.scale_of(spec), T, config.TILE_OVERLAP,
                                 make_infer(self.get(spec)), progress)


_engine: SREngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> SREngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            backend = config.BACKEND
            if backend == "auto":
                backend = "torch" if config.DEVICE != "cpu" and _torch_cuda() else "onnx"
            if backend == "torch":
                _engine = TorchEngine("cuda" if config.DEVICE != "cpu" and _torch_cuda() else "cpu")
            else:
                _engine = OnnxEngine(force_cpu=config.DEVICE == "cpu")
            log.info("SR backend: %s", _engine.device_info())
        return _engine
