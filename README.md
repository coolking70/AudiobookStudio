# OmniVoice Reader Studio

一个面向中文小说、对白和有声书工作流的多人配音工作台。

它把文本分析、角色整理、声音库管理和 OmniVoice 音频合成放在同一个 Web 界面里，适合本地实验、个人工作流和原型验证。

## 主要功能

- 文本分析：调用 OpenAI 兼容接口，把原文切成 `speaker / text / emotion / style`
- 长文本处理：支持分块分析、失败继续、导出已完成 JSON
- 外部 Chat AI 工作流：生成固定提示词模板，把结果再导回系统
- 角色整理：支持分段编辑、角色合并、角色声音分配
- 声音库：管理 `name / style / ref_audio / ref_text`
- 音频生成：逐段调用 OmniVoice 合成，合并 WAV，并生成 LRC

## 项目结构

```text
.
├─ app.py              # FastAPI 入口与接口
├─ pipeline.py         # OmniVoice 封装与逐段合成
├─ role_analyzer.py    # LLM 分析提示词与结果清洗
├─ llm_client.py       # OpenAI 兼容接口客户端
├─ audio_utils.py      # wav 合并与 lrc 生成
├─ schemas.py          # Pydantic 请求模型
├─ static/index.html   # 单页前端
├─ start_local.py      # 跨平台本地启动脚本
├─ start_local.bat     # Windows 辅助启动脚本
├─ start_local.ps1     # Windows PowerShell 启动脚本
├─ start_local.sh      # macOS / Linux 辅助启动脚本
└─ outputs/            # 运行时输出目录
```

## 运行模式

项目支持三种常见运行方式：

1. 仅文本分析
2. 文本分析 + 远程音频后端
3. 文本分析 + 本地 OmniVoice 音频生成

为了提高通用性，当前代码已经改成“按需加载本地音频依赖”：

- 只使用文本分析时，不需要安装 `torch / torchaudio / omnivoice`
- 只有真正调用本地 TTS 时，才会检查 OmniVoice 相关依赖
- 即使缺少本地音频依赖，Web 服务和前端也可以先正常启动

## 运行环境

- Python 3.10+
- Windows / macOS / Linux
- 一个 OpenAI 兼容的 LLM 服务
- 如需本地音频生成，需要额外安装 OmniVoice 运行环境

## 安装

先安装基础依赖：

```bash
pip install -r requirements.txt
```

如果你要实际运行本地 OmniVoice，还需要补装运行时依赖，例如：

```bash
pip install torch torchaudio
```

`omnivoice` 本体和模型缓存的安装方式，取决于你本地验证过的版本与平台，请按自己的环境安装。

## 启动

### 推荐方式

跨平台推荐直接使用：

```bash
python start_local.py
```

可选环境变量：

- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：监听端口，默认 `8000`
- `RELOAD`：设置为 `1/true/yes/on` 时启用热重载
- `AUDIOBOOKSTUDIO_PYTHON`：显式指定启动所用的 Python 解释器

### macOS / Linux

如果你更习惯脚本启动，也可以用：

```bash
./start_local.sh
```

`start_local.sh` 会优先使用 `.venv/bin/python`，否则退回系统里的 `python3` 或 `python`。

### Windows

Windows 下也可以直接运行：

```bat
start_local.bat
```

它会优先使用 `.venv\\Scripts\\python.exe`，否则退回系统里的 `python`。

如果你习惯 PowerShell，也可以运行：

```powershell
.\start_local.ps1
```

Windows 脚本会优先尝试以下解释器来源：

1. `AUDIOBOOKSTUDIO_PYTHON`
2. 当前激活的 `CONDA_PREFIX`
3. `I:\conda_envs\omnivoice\python.exe`
4. `.venv\Scripts\python.exe`
5. 系统 `python`

### 直接使用 uvicorn

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 配置兼容性

### LLM 默认配置文件

项目会按以下顺序寻找 `ai-api.md`：

1. 环境变量 `AUDIOBOOKSTUDIO_LLM_DEFAULTS`
2. 项目根目录下的 `ai-api.md`
3. 项目根目录下的 `config/ai-api.md`
4. 旧版调试路径 `../tts_text_parse/ai-helper/ai-api.md`

如果都不存在，前端仍然可以启动，默认 LLM 配置会留空。

### OmniVoice 模型路径

项目会按以下顺序寻找模型：

1. 环境变量 `OMNIVOICE_MODEL_PATH`
2. Hugging Face 缓存中的 `k2-fsa/OmniVoice`
3. 直接使用模型仓库名 `k2-fsa/OmniVoice`

### 推理设备

项目会按以下顺序选择设备：

1. 环境变量 `OMNIVOICE_DEVICE`
2. CUDA
3. Apple MPS
4. CPU

如需手动指定精度，可设置 `OMNIVOICE_DTYPE`，例如 `float16`、`float32`、`bfloat16`。

## 前端工作流

### 1. 文本分析

- 填写 LLM 配置
- 导入原文或直接粘贴
- 点击“自动分析角色”
- 长文本中断后可继续分析
- 可导出当前已完成 JSON

### 2. 音频生成

- 导入分析结果，或从文本分析页跳转
- 为角色分配声音
- 可按角色随机短句试听
- 最后开始整段音频生成

### 3. 声音库

- 管理声音名称、style、ref_audio、ref_text
- 支持本地音频文件上传
- 支持从声音库复用到角色声音分配

## API 概览

### `GET /`

返回前端页面。

### `GET /api/health`

健康检查。

返回示例：

```json
{
  "ok": true,
  "app": "OmniVoice Reader Studio",
  "model_loaded": false,
  "loaded_devices": [],
  "default_device": "cpu",
  "audio_runtime": {
    "ready": false,
    "missing": [
      "omnivoice"
    ]
  },
  "outputs_dir": "I:\\code\\aitts\\omnivoice-reader\\outputs"
}
```

### `GET /api/llm-defaults`

返回前端默认 LLM 配置。若找不到默认配置文件，也会返回一个空配置结构而不是直接报错。

### `POST /api/parse`

输入原文与 LLM 配置，返回结构化分段。

### `POST /api/segment-plan`

对长文本做智能分块规划。

### `POST /api/optimize-chunks`

对已有文本块进行优化整理。

### `POST /api/analyze-chunks`

对一组文本块发起结构化分析。

### `POST /api/import-segments`

导入已存在的 JSON 分段结果。

### `POST /api/test-connectivity`

测试当前 LLM 服务是否可连通。

### `POST /api/tts`

单段音频生成。

### `POST /api/narrate`

按已有 segments 完成整段合成。

### `POST /api/auto-narrate`

先分析文本，再自动完成整段合成。

### `POST /api/merge`

把已生成的 wav 文件合并为完整音频，并可选生成 LRC。

### `POST /api/upload-ref-audio`

上传声音库参考音频文件，保存到 `outputs/ref_audio/`。

### `GET /api/model/status`

查看当前本地音频模型缓存和已加载设备。

### `GET /api/system/status`

查看平台、CPU、内存和 GPU 状态。若未安装 `psutil` 或 `torch`，会返回降级信息而不是让服务启动失败。

## 已知限制

- `requirements.txt` 只包含基础依赖；OmniVoice 和深度学习框架仍需要按本地环境单独安装
- 一些本地 OpenAI 兼容服务并不完整支持 `response_format` 或多模态输入
- 长文本结构化输出仍然受模型稳定性和 `max_tokens` 影响
- 当前没有自动化测试

## 建议的后续优化

- 为后端补单元测试与接口测试
- 把前端单文件拆成模块化结构
- 为 OmniVoice 模型加载增加更细的设备和缓存控制
- 为音频生成增加后台任务与进度持久化
