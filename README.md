# AI 图片高清化 & 尺寸标准化（AIClear）

高清化家具等商品的产品图片。

批量上传模糊、低清图片 → AI 超分辨率高清化 → 按尺寸模板（宽度 + 宽高比）输出标准规格图片。
全部在本机处理，图片不上传任何第三方；不改动原图主体、构图和色调，看不清的小文字不让 AI 乱补。

- **下载即用**：到 [Releases](https://github.com/Fimobius/AIclearnessFF/releases) 下载
  - Windows 10 / 11（64 位）：`AIClear-v*-Windows-x64.zip`，解压后双击 `Start AIClear.bat`
  - macOS 13.4+：Apple 芯片 `AIClear-v*-macOS-arm64.zip`，Intel `AIClear-v*-macOS-x64.zip`，解压后按住 Control 点击 `Start AIClear.command` → 打开（详见 [第 7 节](#7-macos-版)）
- 面向：素材运营、美工；网页界面 + REST API

---

## 目录

1. [快速开始（使用者）](#1-快速开始使用者)
2. [功能与 PRD 对照](#2-功能与-prd-对照)
3. [效果与验证](#3-效果与验证)
4. [技术方案](#4-技术方案)
5. [需求分析（规划阶段）](#5-需求分析规划阶段)
6. [开发者指南](#6-开发者指南)
7. [macOS 版](#7-macos-版)
8. [第三方组件与许可](#8-第三方组件与许可)

---

## 1. 快速开始（使用者）

1. 下载 Release 里的 `AIClear-v1.0.0-Windows-x64.zip`（约 230 MB），**完整解压**到任意文件夹（路径可含中文、空格；不要放 `C:\Program Files`）
2. 双击 `Start AIClear.bat`，几秒后浏览器自动打开操作页面（地址显示在黑色窗口里，一般是 <http://127.0.0.1:8000>）
3. 拖入图片 → 勾选 → 设置档位、尺寸、缩放方式、导出格式 → 开始处理 → 对比 → 下载
4. 关闭黑色窗口即退出

首次运行若出现「Windows 已保护你的电脑」，点「更多信息 → 仍要运行」。刚解压后的第一次启动，系统会对程序文件做安全扫描，可能要等 1–3 分钟；之后启动只需几秒。

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 1903+ / Windows 11，64 位 |
| 内存 | 8 GB 以上（推荐 16 GB） |
| 显卡 | 任意支持 DirectX 12 的显卡（NVIDIA / AMD / Intel，含集成显卡）自动加速；没有时自动用 CPU |
| 其他 | 不需要安装 Python、CUDA、VC++ 运行库，不需要联网 |

速度参考（输出 1200×1200）：RTX 5070 Laptop 约 1–4 秒 / 张；纯 CPU 约 10–30 秒 / 张。

数据（上传原图、结果、历史）在程序目录的 `data` 文件夹（不可写时在 `%LOCALAPPDATA%\AIClear\data`），默认保留 30 天。

---

## 2. 功能与 PRD 对照

| PRD 要求 | 实现 |
|---|---|
| 单张 / 批量 / 拖拽上传，JPG PNG WEBP | 支持；整批拖入，分批上传带进度条 |
| 大小、数量上限，损坏和非图片文件拦截 | 单张 30 MB、每批 500 张、总量 2 GB；按文件内容（不是扩展名）校验，坏图标红并说明原因 |
| 上传后预览、勾选要处理的图片 | 缩略图网格 + 勾选 / 全选 + 查看原图；右侧实时预览所选图片在目标画布中的构图 |
| 标准增强 / 强力修复两档 | 标准 = SwinIR-M PSNR（保真优先）；强力 = SwinIR-M GAN（细节优先），选型见 [3.1](#31-模型选型) |
| 不改变物体、构图、色彩 | 小波色彩锁定（低频色调取自原图）+ 每张自动保真校验（SSIM、ΔE2000），超阈值标记「需复核」 |
| 避免 AI 脑补、失真 | 小文字保护（默认开启）：原图中太小或太模糊的文字保留原图笔画，不生成错字，见 [3.2](#32-小文字保护) |
| 尺寸模板（宽度 + 宽高比 10:N），一键切换，可保存 | 内置 4 个模板 + 自定义保存；高度 = 宽 × N ÷ 10，四舍五入；除不尽时提示并给出能整除的宽度；也可直接填宽 × 高 |
| 等比补边 / 居中裁剪 / 拉伸（可关闭） | 三种都有；补边底色白 / 黑 / 透明 / 任意色；裁剪可「以商品主体为中心」；可选「去掉原图四周纯色留边」；拉伸可用 `AICLEAR_ALLOW_STRETCH=0` 关闭 |
| 输出格式、JPG 画质、命名规则 | JPG / PNG / WEBP，画质 0–100，文件大小上限（KB），后缀命名 `原名_hd.jpg`，重名自动编号；统一嵌入 sRGB 色彩配置 |
| 任务队列、实时进度、失败重试 | 每张显示阶段、进度、预计耗时；失败单独标红，可单张 / 批量重试；可取消排队 |
| 原图 vs 结果对比 | 全屏对比：滑动分割 / 左右并排 / 100% 像素放大 / 显示文字保护区域，方向键切换，显示保真指标 |
| 单张下载 / ZIP 批量下载 | 都支持；ZIP 附「处理报告.csv」（Excel 直接打开，中文文件名不乱码） |
| 预估耗时、异步处理 | 每张显示预估秒数，按实测速度自动校准；后台队列处理，关掉浏览器不影响 |
| 几十 / 上百张不崩溃、不中断 | 任务状态存 SQLite；进程被强制结束后重启自动续跑，见 [3.4](#34-稳定性) |
| API 接口 | REST API，OpenAPI 文档 `/docs`；可设 `AICLEAR_API_KEY` 开启访问密钥 |
| 兼容性：输出在 Windows、Mac 正常打开 | 标准 JPG / PNG / WEBP + sRGB；自动化测试逐张打开校验 |
| 扩展：底色自定义、文件大小上限、历史记录 | 已做 |
| 扩展：水印、账号权限 | 未做（第二期）；「去除水印」有版权风险，建议不做 |

PRD 修订版把尺寸模板从「宽 × 高」改为「宽度 + 宽高比 10:××」，已按修订版实现（同时兼容直接填宽 × 高）。

---

## 3. 效果与验证

完整方法与数据见 [eval/README.md](eval/README.md)。

### 3.1 模型选型

黄金测试集：12 张 DIV2K 公开基准照片 + 5 张合成商品图（中英文标签、条形码、瓶身文字、织物纹理、电子产品细网格、12 色色卡），每张做 4 种退化（压缩小图、失焦、暗光噪点、重度老旧）并缩小 4 倍，共 68 个样本。每个候选模型放大回原尺寸，与原图比较。

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | ΔE2000↓ |
|---|---|---|---|---|
| Lanczos（无 AI 基线） | 23.39 | 0.607 | 0.630 | 5.48 |
| RealESRNet x4plus | 24.35 | 0.667 | 0.438 | 4.67 |
| RealESRGAN x4plus | 23.22 | 0.618 | 0.315 | 5.13 |
| general-x4v3 降噪 0.5 | 23.64 | 0.630 | 0.394 | 5.07 |
| **SwinIR-M PSNR → 标准增强** | **24.69** | **0.672** | 0.430 | **4.50** |
| **SwinIR-M GAN → 强力修复** | 23.58 | 0.630 | **0.310** | 4.99 |

- 标准增强：所有候选中保真度最高、色差最小、生成的假细节最少
- 强力修复：感知质量最好，四项指标都优于常用的 Real-ESRGAN x4plus
- 色彩锁定在所有模型上都提升 PSNR（约 +0.4 dB）、降低色差

视觉对比：[eval/report/model_montage.png](eval/report/model_montage.png)

### 3.2 小文字保护

**问题**：原图里看不清的小字，任何 AI 模型都还原不了；GAN 模型还会画出「像字但不是字」的笔画。例如条形码数字 `6 901234 567892` 被强力修复改成了 `6 001304 567692`。自动保真校验抓不到这种情况。

**做法**：
1. PP-OCRv3 文字检测，在 AI 的输入图和输出图上各检测一次（后者抓住 AI 凭空画出的字）
2. 以 AI 实际看到的输入为准，满足任一条件就保护：
   - 太小：检测框高度低于阈值（标准 11 px / 强力 13 px）
   - 太模糊：边缘宽度 ÷ 文字高度 > 0.25
3. 保护区域换成普通 Lanczos 缩放（不会产生新笔画），边缘羽化

**阈值用 OCR 外部裁判标定**：8 种字高 × 3 种退化，每种字高约 90 个字符串，统计「读对率 / 读成别的字的比例」（三种退化合计）：

| 原图字高 | 普通缩放 | 标准增强 | 标准+保护 | 强力修复 | 强力+保护 |
|---|---|---|---|---|---|
| 4 px | 0.00/0.07 | 0.00/0.14 | 0.00/0.13 | 0.00/0.26 | 0.00/0.10 |
| 6 px | 0.02/0.60 | 0.08/0.64 | 0.02/0.61 | 0.01/0.77 | 0.02/0.61 |
| 8 px | 0.43/0.51 | 0.56/0.42 | 0.41/0.55 | 0.41/0.56 | 0.36/0.56 |
| 10 px | 0.85/0.13 | 0.84/0.16 | 0.82/0.18 | 0.78/0.21 | 0.78/0.22 |
| 12 px | 0.80/0.20 | 0.93/0.07 | 0.93/0.07 | 0.92/0.07 | 0.92/0.07 |
| 16 px | 0.95/0.05 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 |

结论：
- ≤ 6 px 的字原图本身不可读，AI 只会更多地写错（强力修复尤其明显），保护后回到普通缩放水平
- ≥ 10–12 px 时 AI 读得比普通缩放更准，不保护
- 代价：少数 8 px 左右的字会偏软——宁可模糊，不可写错

视觉对比：[eval/report/text_protection.png](eval/report/text_protection.png)。可在界面取消勾选，或 API 传 `"text_protect": false`。

### 3.3 速度

| 场景（RTX 5070 Laptop） | 每张耗时 |
|---|---|
| 300 px → 1200×1200 | 约 1.0 秒 |
| 600–2000 px → 1200×1200 | 约 3.5 秒（文字保护另加 0.3–0.4 秒） |
| 首张（加载模型） | 另加 5–10 秒 |
| 纯 CPU（无可用显卡） | 约 10–30 秒 |

优化记录：
- 瓦片均分：修正固定 512 瓦片的重复计算，12.1 s → 3.7 s
- SwinIR 用 bf16：再快 1.4 倍，与 fp32 差 44–54 dB PSNR，肉眼不可见
- 色彩锁定移到 GPU：1.12 s → 0.03 s，结果逐像素一致
- CPU 模式关闭 2 倍超采样：计算量减为 1/4

### 3.4 稳定性

`eval/stress_test.py`：200 个文件，其中 190 张有效（含 CMYK、透明 PNG、16 位 PNG、EXIF 旋转）、10 个坏文件。处理到第 67 张时**强制结束服务进程**再重启：

| 检查项 | 结果 |
|---|---|
| 坏文件识别 | 10 / 10 |
| 处理成功 | 190 / 190，失败 0 |
| 强制结束后续跑 | 自动继续，无丢失、无重复 |
| 输出尺寸精确 | 190 / 190 |
| ZIP | 190 张，文件名无重复 |

自动化测试：`backend/tests` 共 84 项。覆盖每种缩放模式 × 档位 × 比例的精确尺寸、色彩配置转换、损坏文件、透明、文件大小上限、ZIP 中文名、重试、模板增删改、文字保护、指标与 scikit-image 一致性。PyTorch 与 ONNX（DirectML / CPU）两种后端都跑过。

---

## 4. 技术方案

### 4.1 路线选择

| 路线 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| A. 本地 GPU + 开源超分模型 | 不按张付费；图片不出本机；可精细控制失真 | 需要选模型、配环境 | ✅ 采用 |
| B. 云 API（阿里云 / 腾讯云 / 百度等） | 不要 GPU | 按张收费；图片外传；脑补不可控；厂商绑定 | 可作为备用后端，未做 |
| C. 扩散模型超分（SUPIR / SeeSR / OSEDiff） | 细节最丰富 | 大量脑补，违反 PRD 核心约束；8 GB 显存吃紧；慢 | ❌ |
| D. 商业软件（Topaz 等） | 效果好 | 授权费；难做网页和 API | ❌ |
| 大模型「智能体」 | — | 核心是确定性图像处理，用不上 | 不进主流程 |

**发行版的推理方式**：开发用 PyTorch + CUDA；发行版把模型导出为 ONNX，用 ONNX Runtime + DirectML 运行。这样做的好处：
- NVIDIA / AMD / Intel 显卡都能加速，没有显卡自动退回 CPU
- 不依赖 CUDA，安装包从约 3 GB 降到约 230 MB
- 与 PyTorch 输出逐像素一致（PSNR 120 dB）

### 4.2 每张图的处理流程

1. **解码校验**：按内容识别格式；损坏、超大图拦截
2. **规范化**：EXIF 方向、ICC 色彩配置转 sRGB（含 CMYK）、16 位转 8 位、透明通道分离
3. **构图规划**：按模板算目标宽高；可选去纯色留边；裁剪模式先在原图上定裁剪框（只对保留部分做 AI，省算力），可识别商品主体位置
4. **AI 超分**：按需放大倍数决定模型输入大小，输出约为目标的 2 倍再缩小（超采样，抑制伪影）；均分瓦片、羽化拼接；显存不足自动减小瓦片
5. **色彩锁定**：小波分解，低频（色调）取原图，高频（细节）取模型
6. **小文字保护**：见 3.2
7. **保真校验**：缩回原尺寸与原图比 SSIM、CIEDE2000 色差，超阈值标「需复核」
8. **合成画布**：补边 / 裁剪 / 拉伸；底色或透明；同时生成同构图的普通缩放图供对比
9. **编码输出**：JPG（≥90 画质用 4:4:4）/ PNG / WEBP，嵌入 sRGB；按大小上限二分搜索画质；逐张断言输出尺寸与模板一致

### 4.3 架构

```
浏览器（React + TypeScript）
   │  REST / 轮询
FastAPI ── SQLite（任务、图片状态、模板，WAL）
   │
后台 GPU 队列（单线程，崩溃 / 重启自动续跑，实测速度校准预估时间）
   │
图像流水线 ── 超分引擎：PyTorch(CUDA) 或 ONNX Runtime(DirectML / CPU)
             文字检测：OpenCV DNN + PP-OCRv3
```

### 4.4 目录

```
backend/app/imaging/  io（解码 / 色彩管理 / 编码）sizing（尺寸与构图）sr（超分引擎）
                      colorfix（色彩锁定）textguard（小文字保护）metrics（SSIM / ΔE）
                      analyze（诊断与保真校验）pipeline（整条流水线）
backend/app/          main（API）worker（GPU 队列）db（SQLite）config（配置）
backend/tests/        自动化测试
frontend/src/         网页界面
eval/                 黄金测试集生成、模型对比、文字评测、压力测试及报告
scripts/              下载模型、导出 ONNX、打包发行版、发行版冒烟测试
packaging/            发行版启动器与说明
models/               模型清单、许可证、校验和（权重文件不入库）
```

---

## 5. 需求分析（规划阶段）

### 5.1 PRD 中发现并处理的问题

1. **前后矛盾**：「✅ 图片抠图、背景替换、AI 绘图生成新图」列在「做」里，与「仅做清晰度增强，不改动原图主体」冲突。按「不做」处理
2. **「去除水印」**：去掉他人图片水印有版权风险，建议只做「添加水印」
3. **验收标准未量化**：「清晰度明显提升」「无严重失真」容易扯皮。已量化：尺寸 100% 精确、色差 ΔE、结构保真 SSIM、盲测偏好率、压测零崩溃
4. **部署方式未写**：按本机单人使用实现；需要多人共用时用 API Key 开放
5. **修订版**：尺寸模板改为「宽度 + 宽高比 10:N」。新写法带来「高度除不尽」的问题，按四舍五入处理并实时提示

### 5.2 四个问题

**① 顶尖专家会额外做什么**
- 按需放大：先算需要放大多少倍，原图够大就不硬放大，避免无谓脑补
- 自动防失真检查：缩回原尺寸与原图比结构和色差，不达标标给人工复核
- 色彩锁定：低频色调取原图，从机制上保证不改色
- 输入处理干净：EXIF 方向、色彩配置、CMYK、透明、16 位、超大图
- 商品图上的小字和 Logo 单独测试并保护
- 补边前去白边、裁剪时识别主体
- 核验模型权重的商用许可
- 每张图可追溯（模型、参数、指标、耗时）
- 先建黄金测试集再选模型

**② 原始材料**
- 业务方真实的模糊图，按模糊成因分类
- 同一张图的高清原版（做有标准答案的测试）
- 目标平台官方图片规范原文
- 模型原始论文、官方权重和 LICENSE
- 美工对「过度 AI 化」的真实判断

**③ 专业方法与工具**
- 超分模型：Real-ESRGAN 系列、SwinIR、HAT；用 spandrel 加载，OpenModelDB 查许可
- 推理：PyTorch CUDA（fp16 / bf16）、分块推理，发行版用 ONNX Runtime DirectML
- 色彩：小波色彩校正、ICC 转换、CIEDE2000
- 质量评估：PSNR / SSIM / LPIPS（有参考）、OCR 文字评测
- 退化合成：Real-ESRGAN 式退化
- 工程：pytest 回归测试、压力测试、端到端测试

**④ 外部验证标准**
- 公开基准数据集上的客观指标：DIV2K，LPIPS 优于基线
- 色差感知阈值：ΔE2000 ≈ 2.3 为人眼刚能察觉
- 主观评测：参照 ITU-R BT.500 / ITU-T P.910 做盲测
- 平台规范：淘宝 / 京东主图 800×800 以上；亚马逊主图最长边 ≥ 1000 px、纯白底。以官方最新文档为准
- 文件兼容性：Windows / macOS 原生看图软件可打开
- 尺寸由程序逐张断言

### 5.3 改写后的 Prompt

> 你是图像复原与超分辨率领域的资深工程师。请根据附件 PRD《AI 图片高清化 & 尺寸标准化智能体》（V1.0 修订版）开发一个可在本机运行的 Web 应用 + REST API。「智能体」不是硬性要求，以稳定、可验证地满足 PRD 为准。
>
> **约束**：
> 1. 保真优先于「看起来更清晰」：禁止改变主体、构图和色调，默认不用扩散类生成式超分
> 2. 只用允许商用的模型权重，并逐个记录许可证
> 3. PRD 中「抠图 / 背景替换 / AI 绘图」与「不改动主体」矛盾，按「不做」处理
>
> **必须做到**：
> - 尺寸模板 = 宽度 + 宽高比（默认 10:N，N 可为小数，兼容 a:b 写法）。高度由程序四舍五入计算并实时显示，除不尽时提示；可保存多套模板；也可直接输入宽 × 高
> - 自适应放大倍数
> - 统一色彩空间、EXIF 方向和透明通道
> - 色彩锁定，并用 SSIM 和 ΔE2000 自动检查失真
> - 小文字保护，不生成错字
> - 三种尺寸模式
> - 任务队列支持断点续跑；失败可重试；ZIP 下载时中文文件名不乱码；提供原图对比和处理报告
>
> **流程**：先用真实样图和合成退化图建立黄金测试集；对比 Real-ESRGAN / SwinIR / HAT 等模型并出报告，由我确定两档模型；再开发后端、前端、API。
>
> **验收（外部标准）**：
> - 尺寸 100% 一致
> - 平均 ΔE2000 < 3
> - SSIM ≥ 阈值（阈值由测试集标定）
> - 文字 OCR 读对率不低于普通缩放
> - 美工盲测偏好率 ≥ 90%，严重失真 ≤ 2%
> - 200 张压测零崩溃
> - 输出文件 Windows / macOS 都能打开
> - 平台尺寸规范以官方文档为准
>
> 先给出规划，经我同意后再开发；每个里程碑给出可验证的证据。

---

## 6. 开发者指南

### 6.1 搭建开发环境（Windows）

```bash
# Python 3.12 虚拟环境
python -m venv backend/.venv
# PyTorch（RTX 50 系必须用 CUDA 12.8 版本）
backend/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt onnx onnxscript onnxruntime-directml
# 模型权重（官方来源 + 镜像回退，SHA-256 校验）
backend/.venv/Scripts/python scripts/download_models.py
# 导出 ONNX（发行版和 onnx 后端使用）
backend/.venv/Scripts/python scripts/export_onnx.py
# 前端
cd frontend && npm install && npm run build
```

启动开发服务（默认 PyTorch CUDA 后端；`AICLEAR_BACKEND=onnx` 切到发行版后端）：

```bash
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000
```

前端热更新：`cd frontend && npm run dev`（5173 端口，API 自动代理到 8000）。

### 6.2 测试与评估

```bash
cd backend && .venv/Scripts/python -m pytest tests -q          # 自动化测试
AICLEAR_BACKEND=onnx .venv/Scripts/python -m pytest tests -q   # 发行版后端
cd .. && backend/.venv/Scripts/python eval/make_golden.py      # 生成黄金测试集（DIV2K 原图需自行下载到 eval/golden/hr_div2k）
backend/.venv/Scripts/python eval/bakeoff.py                   # 模型对比 → eval/report/index.html
backend/.venv/Scripts/python scripts/download_models.py --eval # OCR 评测模型
backend/.venv/Scripts/python eval/text_eval.py                 # 文字保护标定
backend/.venv/Scripts/python eval/stress_test.py --n 200       # 压力 + 崩溃续跑测试
```

### 6.3 打包发行版

```bash
backend/.venv/Scripts/python scripts/build_release.py                 # 三个平台都打包
backend/.venv/Scripts/python scripts/build_release.py macos-arm64     # 或只打一个：windows-x64 / macos-arm64 / macos-x64
# → build/AIClear-v<版本>-{Windows-x64, macOS-arm64, macOS-x64}.zip
backend/.venv/Scripts/python scripts/test_package.py <解压后的 AIClear 目录> [--cpu]   # 运行包内自检
backend/.venv/Scripts/python scripts/inspect_mac_package.py build/AIClear-v<版本>-macOS-arm64.zip arm64
```

Mac 包可以在 Windows 上组装：Python 运行时与 macOS 依赖都是现成的二进制，脚本最后逐个检查 Mach-O 文件的架构和最低系统版本。

发行版内容：
- 可移动的 CPython 3.12（python-build-standalone）与依赖
- 随程序附带的 MSVC 运行库（app-local，用户无需安装 VC++ 运行库）
- 后端代码、构建好的网页
- 两个 SwinIR ONNX 模型和文字检测模型
- 启动器：
  - 自动选空闲端口
  - 端口记录在 `data/server.json`，第二次启动直接打开已运行的实例
  - 自动打开浏览器
- 包内自检 `selftest.py`（Mac 上双击 `Run Self-Test.command`）

冒烟测试做法：
- 只留系统目录在 PATH 上，确认不依赖开发环境
- 放在含中文和空格的路径下运行
- 分别测试 GPU 和 CPU 模式

### 6.4 API 最简用法

```bash
curl -F "files=@a.jpg" -F "files=@b.png" -F "start=true" \
     -F 'settings={"tier":"standard","width":1200,"ratio_w":10,"ratio_h":13,"mode":"pad","format":"jpg","quality":92}' \
     http://127.0.0.1:8000/api/jobs
# 返回 job；轮询 GET /api/jobs/{id}，完成后 GET /api/jobs/{id}/download 下载 ZIP
```

完整接口文档：服务启动后访问 `/docs`。

### 6.5 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AICLEAR_BACKEND` | auto | auto（有 CUDA 用 torch，否则 onnx）/ torch / onnx |
| `AICLEAR_DEVICE` | auto | cpu = 强制 CPU |
| `AICLEAR_API_KEY` | 空 | 设置后所有接口需带 `X-API-Key` |
| `AICLEAR_ALLOW_STRETCH` | 1 | 0 = 关闭拉伸模式 |
| `AICLEAR_MAX_FILE_MB` / `MAX_FILES_PER_JOB` / `MAX_JOB_MB` | 30 / 500 / 2048 | 上传限制 |
| `AICLEAR_MAX_SIDE` / `MAX_OUTPUT_PIXELS` | 10000 / 40000000 | 输出尺寸上限 |
| `AICLEAR_QA_MAX_DELTA_E` / `QA_MIN_SSIM` | 3.0 / 0.85 | 「需复核」阈值 |
| `AICLEAR_RETENTION_DAYS` | 30 | 历史保留天数，0 = 永久 |
| `AICLEAR_HOST` | 127.0.0.1 | 发行版启动器监听地址；设为 `0.0.0.0` 可共享给局域网（必须同时设置 `AICLEAR_API_KEY`） |
| `AICLEAR_DATA_DIR` / `MODELS_DIR` / `FRONTEND_DIST` | 项目内 | 路径 |

---

## 7. macOS 版

**状态：v1.1.0 已实现并打包（Apple 芯片 / Intel 各一个），本机能做的验证全部通过，等待 Mac 真机验收。** 系统要求 **macOS 13.4 Ventura 或更新**。

### 7.1 使用方法（Mac）

1. 按芯片下载：Apple 芯片（M1–M4）选 `AIClear-v*-macOS-arm64.zip`，Intel 选 `AIClear-v*-macOS-x64.zip`
2. 双击解压，**按住 Control 键点击 `Start AIClear.command` → 打开**（本程序未做 Apple 签名，第一次需要这样打开）
3. 浏览器自动打开操作页面；关闭终端窗口即退出
4. 数据在 `~/Library/Application Support/AIClear/data`
5. 双击 `Run Self-Test.command` 可自检，生成 `selftest_report.txt`

### 7.2 与 Windows 版的差异

| 方面 | Windows | macOS |
|---|---|---|
| 推理加速 | ONNX Runtime + DirectML（任意 DX12 显卡） | ONNX Runtime + CoreML（Apple GPU / 神经网络引擎） |
| 依赖版本 | ONNX Runtime 1.24.4、OpenCV 5.0、NumPy 2.5 | ONNX Runtime 1.23.2、OpenCV 4.12、NumPy 2.2（支持 macOS 13 的最新版本） |
| 启动 | `Start AIClear.bat` | `Start AIClear.command`（首次打开后自动清除本文件夹的下载隔离标记） |
| 数据位置 | 程序目录 `data`（不可写时 `%LOCALAPPDATA%`） | `~/Library/Application Support/AIClear`（不写入程序目录） |
| 运行库 | 自带 MSVC 运行库 | 不需要 |

模型、算法、网页、文字保护、所有功能完全相同。

### 7.3 关键技术工作

**1. 让 SwinIR 能整块跑在 CoreML 上**

用 ONNX Runtime 官方的分析工具检查，原始导出的 SwinIR 会被 CoreML 切成 **111 块**，在 CoreML 和 CPU 之间来回切换。工具判断「很可能比纯 CPU 还慢」。原因和处理（只在导出时改写，与原模型数学上完全等价，输出与原始 PyTorch 模型逐像素一致，PSNR 120 dB）：

| 原因 | 处理 |
|---|---|
| `q, k, v = qkv[0], qkv[1], qkv[2]` 导出为标量 Gather，CoreML 不支持 | 改用 `torch.split` |
| 窗口切分 / 还原用 6 维张量，CoreML 最多 5 维 | 固定批大小 1，改写为 5 维 |
| 相对位置偏置、移位窗口掩码在运行时计算 | 按固定 256 瓦片预先算成常量 |
| 形状运算在运行时计算，CoreML 看到的是动态形状 | 用 ONNX Runtime 基础级优化折叠成常量（与执行后端无关，可跨平台） |

改写后：**1 块，1769 / 1769 个节点全部由 CoreML 执行**，工具判断「should perform well: YES」。

另外两点：
- 官方分析工具会把 Resize 的空可选输入误判为「动态形状」，`scripts/coreml_coverage.py` 修正了这一点
- 改写后的模型在 Windows 上也略快：DirectML 0.58 → 0.50 秒 / 瓦片

**2. 首次运行自检（加速是否可信）**

第一次加载每个模型时，同一个瓦片分别在加速后端和 CPU 上各跑一次，只有同时满足以下两点才启用加速，否则自动改用 CPU：
- 输出一致（PSNR ≥ 40 dB）
- 确实更快

结果缓存，只做一次。这一机制同时防止 CoreML 低精度计算出错、Windows 显卡驱动异常等情况。本机 DirectML 实测：PSNR 129.7 dB，0.66 秒对比 CPU 8.1 秒，启用。

**3. 在 Windows 上组装 Mac 包，并逐个检查二进制文件**

- Python 运行时用 python-build-standalone 官方 macOS 版，按 SHA-256 校验
- 依赖用 uv 按 macOS 13 平台下载
- 用 `scripts/macho_minos.py` 读取包内每个 Mach-O 二进制的最低系统版本和 CPU 架构
- 这个检查发现：ONNX Runtime 1.23 的安装包标的是 macOS 13.0，但二进制实际要求 **13.4**（1.20–1.22 实际要求 13.3）。只看安装包标签会在 13.0–13.3 上启动失败，所以最低要求定为 13.4。Ventura 用户可以在「软件更新」里免费升到 13.7
- `scripts/inspect_mac_package.py` 检查：
  - 架构正确（protobuf 为 universal2）
  - 没有 Windows 文件混入
  - 依赖版本与锁定一致
  - 启动脚本可执行且为 LF 换行

### 7.4 已验证与待验证

| 项目 | 状态 |
|---|---|
| 代码与模型在 Mac 版依赖版本上运行（ONNX Runtime 1.23.2 + OpenCV 4.12 + NumPy 2.2） | ✅ 本机全部测试通过 |
| 改写后的模型与原模型一致 | ✅ PSNR 120 dB（PyTorch / DirectML / CPU） |
| CoreML 算子覆盖 | ✅ 100%，1 个分区（静态分析） |
| 包内所有二进制：架构、最低系统版本 | ✅ arm64 / x86_64 全部 ≤ 13.4 |
| 首次运行加速自检的各种失败情况 | ✅ 单元测试：输出不一致、更慢、崩溃、缓存 |
| 自检脚本 `selftest.py` | ✅ 在 Windows 包上通过 |
| CoreML 实际速度与精度 | ⏳ 需 Mac 真机 |
| Gatekeeper / 隔离标记处理、`.command` 启动 | ⏳ 需 Mac 真机 |
| Intel Mac（x64 包） | ⏳ 需 Intel Mac 或其上的 macOS 13 |

### 7.5 真机验收步骤

1. 在 Mac 上下载对应的 zip，解压
2. 按住 Control 点击 `Run Self-Test.command` → 打开，等待显示「自检通过 / 自检失败」
3. 把生成的 `selftest_report.txt` 发回。报告里包括：
   - 系统版本和芯片
   - 选中的后端（CoreML 或 CPU）
   - 加速自检结果（PSNR、加速与 CPU 耗时）
   - 每张测试图的耗时和结果
   - 服务日志
4. 再按 7.1 正常使用一次，确认浏览器、拖拽上传、对比、下载都正常

### 7.6 对外分发（可选）

未签名的程序第一次需要「Control + 点击 → 打开」。如果要让用户直接双击打开，需要：
- 注册 Apple 开发者账号（99 美元 / 年）
- 在 Mac 上用 `codesign` 签名，用 `notarytool` 公证

---

## 版本记录

| 版本 | 内容 |
|---|---|
| v1.1.0 | 新增 macOS 版（Apple 芯片 / Intel，macOS 13.4+，CoreML 加速）<br>SwinIR 改写为对加速器友好的等价形式（CoreML 1 个分区；DirectML 也略快）<br>首次运行自动校验加速后端的正确性和速度，不合格自动改用 CPU<br>修复 Windows 每次启动多等约 100 秒的问题（逐个探测空端口）<br>启动器、自检脚本跨平台 |
| v1.0.0 | 首个 Windows 版：AI 高清化（两档）、尺寸模板、三种缩放、小文字保护、保真校验、批量队列、对比、ZIP、REST API |

---

## 8. 第三方组件与许可

| 组件 | 许可 |
|---|---|
| SwinIR 权重（Jingyun Liang 等） | Apache-2.0 |
| PP-OCRv3 文字检测（PaddlePaddle，经 OpenCV Zoo） | Apache-2.0 |
| Real-ESRGAN 权重（仅评估用） | BSD-3-Clause |
| ONNX Runtime / DirectML | MIT / Microsoft 再分发许可 |
| OpenCV | Apache-2.0 |
| CPython | PSF |
| FastAPI、Pydantic、React | MIT |
| NumPy、Uvicorn | BSD |
| Pillow | MIT-CMU |

模型权重的文件名、来源、SHA-256 见 [models/MODELS.md](models/MODELS.md)。DIV2K 测试图片仅供研究，不随仓库分发。
