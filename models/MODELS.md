# Model weights

All weights come from the authors' official GitHub releases (downloaded through gh-proxy.com; leading bytes cross-checked against direct GitHub downloads). Only licences that allow commercial use are included.

| File | Licence | Source | SHA-256 |
|---|---|---|---|
| `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | Apache-2.0 | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 | `b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6` |
| `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth` | Apache-2.0 | https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0 | `1fd8fed99684bd271db55563e4906d36459cf535446820053f7a1081d4781dc5` |
| `RealESRGAN_x2plus.pth` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` |
| `RealESRGAN_x4plus.pth` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| `RealESRNet_x4plus.pth` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| `realesr-general-wdn-x4v3.pth` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |
| `realesr-general-x4v3.pth` | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| `text_detection_cn_ppocrv3_2023may.onnx`（文字检测，PP-OCRv3，用于小文字保护） | Apache-2.0 | https://huggingface.co/opencv/text_detection_ppocr | `03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587` |

## ONNX models used by the packaged apps

`SwinIR-M_x4_PSNR_t256.onnx` and `SwinIR-M_x4_GAN_t256.onnx` are generated from the two SwinIR
weights above by `scripts/export_onnx.py` (fixed 256x256 tile, batch 1). The export rewrites
SwinIR into a mathematically identical, accelerator-friendly form (no scalar Gather, at most
5-D tensors, pre-computed attention bias / masks, folded shape arithmetic) and verifies the
output against the original PyTorch model (PSNR 120 dB). They ship inside the release zips.
