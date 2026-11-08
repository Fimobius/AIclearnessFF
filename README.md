# AI 图片高清化 & 尺寸标准化（AIClear）

高清化家具等商品的产品图片。

批量上传模糊、低清图片 → AI 超分辨率高清化 → 按尺寸模板（宽度 + 宽高比）输出标准规格图片。
全部在本机处理，图片不上传任何第三方；不改动原图主体、构图和色调，看不清的小文字不让 AI 乱补。

- **下载即用（Windows 10 / 11，64 位）**：到 [Releases](https://github.com/Fimobius/AIclearnessFF/releases) 下载 `AIClear-v*-Windows-x64.zip`，解压后双击 `Start AIClear.bat`
- 面向：素材运营、美工；网页界面 + REST API

---

## 目录

1. [快速开始（使用者）](#1-快速开始使用者)
2. [功能与 PRD 对照](#2-功能与-prd-对照)
3. [效果与验证](#3-效果与验证)
4. [技术方案](#4-技术方案)
5. [需求分析（规划阶段）](#5-需求分析规划阶段)
6. [开发者指南](#6-开发者指南)
7. [macOS 部署分析](#7-macos-部署分析)
8. [第三方组件与许可](#8-第三方组件与许可)

---

## 1. 快速开始（使用者）

1. 下载 Release 里的 `AIClear-v1.0.0-Windows-x64.zip`（约 230 MB），**完整解压**到任意文件夹（路径可含中文、空格；不要放 `C:\Program Files`）
2. 双击 `Start AIClear.bat`，几秒后浏览器自动打开操作页面（地址显示在黑色窗口里，一般是 <http://127.0.0.1:8000>）
3. 拖入图片 → 勾选 → 设置档位、尺寸、缩放方式、导出格式 → 开始处理 → 对比 → 下载
4. 关闭黑色窗口即退出

首次运行若出现「Windows 已保护你的电脑」，点「更多信息 → 仍要运行」。

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
backend/.venv/Scripts/python scripts/build_release.py
# → build/AIClear-v<版本>-Windows-x64.zip
backend/.venv/Scripts/python scripts/test_package.py <解压后的 AIClear 目录> [--cpu]
```

发行版内容：
- 可移动的 CPython 3.12（python-build-standalone）与依赖
- 随程序附带的 MSVC 运行库（app-local，用户无需安装 VC++ 运行库）
- 后端代码、构建好的网页
- 两个 SwinIR ONNX 模型和文字检测模型
- 启动器（自动选空闲端口、检测已运行实例、自动打开浏览器）

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

## 7. macOS 部署分析

**目标系统：macOS 13 Ventura 及以上**（不支持 macOS 12）。

**结论：可以部署，但要单独做一个 Mac 版本，并在真机上验收。**
- 核心算法、模型和网页不用改
- 需要替换 Windows 专属的部分：DirectML 加速、`.bat` 启动、VC++ 运行库、打包脚本
- 依赖要锁定到 macOS 13 能装的版本
- 预计工作量：1–2 天开发，另加真机测试

### 7.1 各组件在 macOS 13 上的情况

以下数据直接查自 PyPI 安装包的平台标签：标签 `macosx_14_0` 的包装不到 13 上。

| 组件 | Windows 版用的版本 | macOS 支持情况 | Mac 版方案 |
|---|---|---|---|
| ONNX Runtime | 1.24.4 | 1.24 起只支持 macOS 14+ 的 Apple 芯片 | **锁定 1.23.2**：最后一个支持 macOS 13 的版本，Intel 和 Apple 芯片都有 |
| OpenCV | 5.0.0.93 | Apple 芯片：5.0 仍支持 13；Intel：4.13 起要求 14 | **锁定 4.12.0.88**：两种芯片都支持 macOS 13 |
| NumPy | 2.5.4 | 有 macOS 13 的包，但 OpenCV 4.12 要求 NumPy < 2.3 | **锁定 2.2.6** |
| Pillow / Pydantic / FastAPI | 当前版本 | 仍提供 10.13（Intel）、11.0（Apple 芯片）起的包 | 不用改 |
| Python 3.12 运行时 | python-build-standalone | 官方说明 Intel ≥ 10.13、Apple 芯片 ≥ 11 | 不用改，需真机确认 |
| PyTorch | 只用于开发 | 新版只支持 macOS 14+ 的 Apple 芯片 | Mac 版不需要（用 ONNX） |
| 网页界面 | 已面向 Safari 15+ 构建 | macOS 13 自带 Safari 16，可升级到 18 | 不用改 |
| 字体 | Bahnschrift（Windows 自带） | macOS 没有 | 已有后备字体 DIN Alternate、PingFang SC |

**已验证**：在本机用 ONNX Runtime 1.23.2 + OpenCV 4.12.0.88 + NumPy 2.2.6（与 Mac 版要锁定的版本相同，Windows 版本的包）跑完全部 84 项测试，全部通过。说明模型文件、文字检测、整条流水线在这组版本上都能工作。

**未验证**（必须在 Mac 上测）：
- CoreML 加速的实际效果
- macOS 运行时本身
- Gatekeeper 安全提示
- 真实性能

### 7.2 需要的改动

1. **推理加速**：Windows 用 DirectML，Mac 改用 ONNX Runtime 自带的 CoreML 后端（`CoreMLExecutionProvider`），可用 Apple GPU 和神经网络引擎，不支持的算子自动回退 CPU。SwinIR 的注意力算子不一定都能在 CoreML 上运行，可能部分回退 CPU，需要实测；改动集中在 `backend/app/imaging/sr.py` 的 `OnnxEngine`
2. **启动方式**：
   - `.bat` 换成 `.command` 脚本，或用 PyInstaller 打成 `.app`
   - 数据目录改为 `~/Library/Application Support/AIClear`
3. **打包脚本**：
   - 不需要 VC++ 运行库，改用 macOS 版 Python 运行时
   - 依赖可以在 Windows 上用 `pip download --platform macosx_13_0_arm64` / `macosx_13_0_x86_64` 下载组装
   - Apple 芯片和 Intel 各出一个包
4. **签名与公证**：
   - 未签名的程序下载后会被 Gatekeeper 拦截，提示「无法验证开发者」。内部使用可以「右键 → 打开」，或执行 `xattr -dr com.apple.quarantine AIClear`
   - 对外分发需要 Apple 开发者账号（99 美元 / 年），在 Mac 上用 `codesign` 签名、`notarytool` 公证，只能在 Mac 上完成

### 7.3 硬件与性能预期

macOS 13 支持 2017 年及以后的 Intel Mac 和所有 Apple 芯片 Mac。

| 机型 | 加速 | 预计速度（1200×1200） |
|---|---|---|
| Apple 芯片（M1 及以后） | CoreML 或 CPU | 预计在 Windows 独显与 CPU 之间，需实测 |
| Intel Mac（2017–2020） | 基本只能 CPU | 预计 15–60 秒 / 张；可再调小超采样 |

### 7.4 测试条件

- GitHub 云端的 macOS 13 机器已停用，现在只提供 macOS 14 以上。云端 CI 可以跑 macOS 14/15 Apple 芯片；macOS 13 和 Intel Mac 需要真机，或在 Mac 上用 UTM / Parallels 装虚拟机
- 验收可直接复用 `backend/tests` 和 `scripts/test_package.py`（启动部分要先改成 macOS 版）

### 7.5 建议路线

| 方案 | 说明 | 适合 |
|---|---|---|
| A. 原生 Mac 版（推荐） | 按 7.2 做 Apple 芯片和 Intel 两个包，锁定 7.1 的版本，未签名先内部用 | 美工在自己的 Mac 上离线使用 |
| B. 一台 Windows 当服务器 | 不做 Mac 版：在有显卡的 Windows 上运行本程序，开启访问密钥后共享到局域网，Mac 用 Safari 访问 | 最快：Mac 上什么都不用装，速度取决于 Windows 机器 |
| C. Docker | macOS 上 Docker 不能用 GPU，只能跑 CPU，并且要装 Docker Desktop | 不推荐 |

如果近期只是少数 Mac 用户，先用方案 B，现有程序已经支持。在 Windows 上新建一个 `.bat`：

```bat
set AICLEAR_HOST=0.0.0.0
set AICLEAR_API_KEY=换成你自己的密钥
call "Start AIClear.bat"
```

启动后黑色窗口会显示局域网地址。Mac 用 Safari 打开该地址，按提示输入密钥即可。需要离线使用或 Mac 用户很多时，再做方案 A。

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
