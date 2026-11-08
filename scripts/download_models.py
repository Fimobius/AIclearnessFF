"""Download the model weights from their official sources and verify SHA-256.

  backend/.venv/Scripts/python scripts/download_models.py          # tier models + text detector
  backend/.venv/Scripts/python scripts/download_models.py --all    # also the bake-off candidates
  backend/.venv/Scripts/python scripts/download_models.py --eval   # also the OCR model for eval/

Each file is tried from the official URL first, then from mirrors (useful where GitHub /
Hugging Face are slow). The hash check guarantees mirrors serve the identical file.
After downloading, run scripts/export_onnx.py to produce the ONNX models used by the app.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH_SWINIR = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
GH_ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/"
HF = "https://huggingface.co/opencv/"

# (target dir, file, url, sha256, group)
FILES = [
    ("models", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth", GH_SWINIR + "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",
     "1fd8fed99684bd271db55563e4906d36459cf535446820053f7a1081d4781dc5", "core"),
    ("models", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth", GH_SWINIR + "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
     "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6", "core"),
    ("models", "text_detection_cn_ppocrv3_2023may.onnx", HF + "text_detection_ppocr/resolve/main/text_detection_cn_ppocrv3_2023may.onnx",
     "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587", "core"),
    ("models", "RealESRGAN_x4plus.pth", GH_ESRGAN + "v0.1.0/RealESRGAN_x4plus.pth",
     "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", "all"),
    ("models", "RealESRNet_x4plus.pth", GH_ESRGAN + "v0.1.1/RealESRNet_x4plus.pth",
     "a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df", "all"),
    ("models", "RealESRGAN_x2plus.pth", GH_ESRGAN + "v0.2.1/RealESRGAN_x2plus.pth",
     "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb", "all"),
    ("models", "realesr-general-x4v3.pth", GH_ESRGAN + "v0.2.5.0/realesr-general-x4v3.pth",
     "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292", "all"),
    ("models", "realesr-general-wdn-x4v3.pth", GH_ESRGAN + "v0.2.5.0/realesr-general-wdn-x4v3.pth",
     "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca", "all"),
    ("eval/ocr", "text_recognition_CRNN_CN_2021nov.onnx", HF + "text_recognition_crnn/resolve/main/text_recognition_CRNN_CN_2021nov.onnx",
     "c760bf82d684b87dfabb288e6c0f92d41a8cd6c1780661ca2c3cd10c2065a9ba", "eval"),
    ("eval/ocr", "charset_32_94_3944.h", HF + "text_recognition_crnn/resolve/main/charset_32_94_3944.h", None, "eval"),
]


def mirrors(url: str) -> list[str]:
    if url.startswith("https://github.com/"):
        return [url, "https://gh-proxy.com/" + url, "https://ghfast.top/" + url]
    if url.startswith("https://huggingface.co/"):
        return [url, url.replace("https://huggingface.co/", "https://hf-mirror.com/")]
    return [url]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path, url: str, digest: str | None) -> None:
    if dest.exists() and (digest is None or sha256(dest) == digest):
        print(f"ok        {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for u in mirrors(url):
        try:
            print(f"download  {dest.name} <- {u}", flush=True)
            with urllib.request.urlopen(u, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if digest and sha256(tmp) != digest:
                print("          checksum mismatch, trying next source")
                continue
            tmp.replace(dest)
            return
        except Exception as e:
            print(f"          failed: {e}")
    tmp.unlink(missing_ok=True)
    sys.exit(f"could not download {dest.name}")


def main() -> None:
    groups = {"core"} | ({"all"} if "--all" in sys.argv else set()) | ({"eval"} if "--eval" in sys.argv else set())
    for d, name, url, digest, group in FILES:
        if group in groups:
            fetch(ROOT / d / name, url, digest)


if __name__ == "__main__":
    main()
