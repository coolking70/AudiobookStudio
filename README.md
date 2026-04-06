# OmniVoice Reader Studio

一个面向中文小说/对白文本的多人配音工作台。

它提供了三条核心链路：

- 文本分析：调用 OpenAI 兼容接口，把原文切成 `speaker / text / emotion / style`
- 声音分配：给识别出的角色分配声音库模板，并快速试听短句
- 音频生成：逐段调用 OmniVoice 合成音频，最终合并为完整 WAV，并同步生成 LRC

当前项目更适合本地实验、个人工作流和小型原型验证，而不是开箱即用的生产部署版本。

## 主要功能

- 支持 `.txt` / `.md` 原文导入
- 支持长文本分块分析、失败后继续分析、导出已完成 JSON
- 支持外部 Chat AI 分析模式
- 支持分段结果编辑、角色合并
- 支持声音库管理、参考音频上传、角色声音分配
- 支持逐段 TTS 合成、LRC 生成、播放器歌词同步滚动

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
└─ outputs/            # 运行时输出目录
```

## 运行环境

- Python 3.10+
- Windows / Linux 均可，当前前端和本地模型工作流在 Windows 上做了较多验证
- 需要一个 OpenAI 兼容的 LLM 服务
- 需要 OmniVoice 运行环境

## 安装

先安装基础依赖：

```bash
pip install -r requirements.txt
```

如果你要实际跑 OmniVoice，还需要安装对应的运行时依赖，例如：

```bash
pip install torch torchaudio
```

`omnivoice` 本体和模型缓存的安装方式，取决于你当前使用的 OmniVoice 版本与运行环境，请按你本地已验证的方式安装。

## 启动

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 前端工作流

### 1. 文本分析

- 填写 LLM 配置
- 导入原文或直接粘贴
- 点击“自动分析角色”
- 若长文本中断，可点击“继续上次分析”
- 可在完成后导出 JSON

### 2. 音频生成

- 导入分析结果，或直接使用分析页跳转过来的结果
- 选择角色排序方式
- 为角色分配声音
- 可先试听随机短句
- 最后点击“开始生成”

### 3. 声音库

- 维护声音名称、style、ref_audio、ref_text
- 支持本地音频文件上传
- 支持从声音库快速复用到角色分配

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
  "outputs_dir": "I:\\code\\aitts\\omnivoice-reader\\outputs"
}
```

### `POST /api/parse`

输入原文与 LLM 配置，返回结构化分段。

### `POST /api/import-segments`

导入已存在的 JSON 分段结果。

### `POST /api/test-connectivity`

测试当前 LLM 服务是否可连通。

### `POST /api/tts`

单段音频生成。

### `POST /api/merge`

把已生成的 wav 文件合并为完整音频，并可选生成 LRC。

### `POST /api/narrate`

按已有 segments 完成整段合成。

### `POST /api/auto-narrate`

先分析文本，再自动完成整段合成。

### `POST /api/upload-ref-audio`

上传声音库参考音频文件，保存到 `outputs/ref_audio/`。

## 这次整理包含的改良

- 增加了输出文件名清洗，避免不安全或不可用的文件名
- 增加 `GET /api/health`
- 抽取了后端公共错误包装与分段文件路径辅助函数
- 补充了 `.gitignore`
- 补充了依赖和运行说明
- 补充了 API 与工作流文档

## 已知限制

- `requirements.txt` 只包含基础依赖；OmniVoice 和深度学习框架仍需要按本地环境单独安装
- 一些本地 OpenAI 兼容服务并不完整支持 `response_format` 或多模态输入
- 长文本结构化输出仍然受模型稳定性和 `max_tokens` 影响
- 当前没有自动化测试

## 建议的后续优化

- 为后端补单元测试与接口测试
- 把前端单文件拆成模块化结构
- 为 OmniVoice 模型加载增加显存 / 设备配置选项
- 为音频生成增加后台任务与进度持久化
