# 权威样本生产流程（说话人归因评测集）

本文档面向**指挥其他 AI 接力生产剩余 seg 样本**。照着做即可从大原文里切出一段、跑模型解析、
人工/AI 复核、固化成权威 groundtruth，并机器验收。

参考样板：**`muli4_seg5`** 是第一套用确定性工具固化的样本，所有口径机器可复现 —— 新样本一律
向它看齐。（`seg1`~`seg4` 是更早的手工记账版本，约定略有出入，见文末「历史样本」。）

---

## 0. 一次性准备：API Key（务必看这段）

⚠️ **本仓库是公开的 GitHub 仓库。绝不要把 API key 明文写进任何会被提交的文件**（包括本文档、
脚本、parse.json）。公开仓库里的 key 会在几分钟内被爬虫扫到盗刷，免费额度也会被服务商封禁。

正确做法 —— key 放在 **gitignore 的 `.env`** 里，只在本地工作副本可见：

```bash
cp .env.example .env          # .env 已在 .gitignore 中，不会被 push
# 编辑 .env，把真实 AGNES_API_KEY 填进去
source .env                   # 之后本终端的脚本都能读到 os.environ["AGNES_API_KEY"]
echo "${AGNES_API_KEY:+key 已就绪}"
```

Agnes 服务参数（脚本里已写死，无需改）：

| 项 | 值 |
|---|---|
| base_url | `https://apihub.agnes-ai.com/v1` |
| model | `agnes-2.0-flash` |
| 接口 | OpenAI 兼容 `/chat/completions` |
| 额度 | 免费、不限量 token |

> 给其他 AI 下指令时，只需说「key 在 `.env`，先 `source .env`」即可，不要把 key 值贴进对话或文件。

---

## 1. 一套样本 = 5 个同名文件

每个 seg 是大原文里一段约 1.9 万字的**顺序切片**，产出 5 个文件（以 seg5 为例）：

| 文件 | 角色 | 由谁生成 | 权威性 |
|---|---|---|---|
| `muli4_seg5_sample.txt` | 原文切片（输入） | 从大原文切出 | 原始输入 |
| `muli4_seg5_parse.json` | 模型解析结果（285 段，含 speaker/confidence/evidence…） | `agnes-2.0-flash` 全流水线 | 模型口径 |
| `muli4_seg5_transcript.txt` | parse 的人类可读视图（每段一行 speaker+conf） | 由 parse 渲染 | 便捷视图，**非权威** |
| `muli4_seg5_review.json` | 人工/AI 复核修正（`{段号: 正确说话人}`） | 复核工具产出 | 人工修正 |
| `muli4_seg5_groundtruth.json` | **权威标准答案** + 准确率口径 | parse+review 确定性固化 | ✅ 评测以此为准 |

---

## 2. 生产流程（seg6 为例，整段照搬即可）

### Step 1 — 切出 `sample.txt`

原文：`docs/samples/muli4_original_125697_utf8.txt`（UTF-16 编码，184,083 字）。
各 seg 是**首尾相接的顺序切片**，每段约 19,000 字。已用区间（字符位）：

| seg | start | end |
|---|---:|---:|
| seg1 | 39 | 19,091 |
| seg2 | 19,091 | 38,143 |
| seg3 | 38,151 | 57,259 |
| seg4 | 57,267 | 76,328 |
| seg5 | 76,336 | 95,405 |
| **seg6** | **≈95,405** | **≈114,500** |

切的时候从上一段结尾接着走，长度凑到约 1.9 万字，并**在自然段（空行）边界收口**，不要把一句话/
一段对话切断。原文还剩约 9 万字，够切到 ~seg9-10。

```python
# 切片示例（在 .venv 里跑）
orig = open("docs/samples/muli4_original_125697_utf8.txt", encoding="utf-16").read()
chunk = orig[95405:114500]              # 粗切
# 往后找最近的空行收口，避免切断段落，再 .strip() 写出
open("docs/samples/muli4_seg6_sample.txt", "w", encoding="utf-8").write(chunk.strip())
```

### Step 2 — 跑模型解析 → `parse.json` + `transcript.txt`

**一条命令搞定，必须用它**（封装了与 seg5 完全一致的完整流水线，别手搓、别裸调模型）：

```bash
source .env                                              # 让 AGNES_API_KEY 就位
.venv/bin/python tools/parse_sample.py --seg muli4_seg6
```

它内部用 `parse_novel(..., enable_block_review=True)` + Agnes 参数
（`batch_size=8, max_tokens=5000, temperature=0, context_chars=320, output_mode="compact",
disable_thinking=True, narrator="甘织玲奈子"`），跑完块级复核/密集场景路由/称呼回查等全部纠错
阶段，再用 `model_dump(mode="json")` 正确序列化，产出 `parse.json`（含 21 字段 + 完整 stats）
和初版 `transcript.txt`。完成后会打印实际跑过的流水线阶段，确认不是裸直出。

> ❌ **绝不要**用「裸 `agnes-2.0-flash` 单遍调用」拼 parse —— 那会缺纠错阶段和字段，
> `model_speaker` 不可比，Step 5 会被溯源校验 FAIL。详见 Step 5 的翻车点说明。

`transcript.txt` 会在 Step 4 由 `build_groundtruth.py` 从最终 parse 再重生成一次，保持同步。

### Step 3 — 人工/AI 复核 → `review.json`

启动复核工具（纯 stdlib 本地网页）：

```bash
.venv/bin/python tools/review_server.py \
  --parse docs/samples/muli4_seg6_parse.json \
  --raw   docs/samples/muli4_seg6_sample.txt \
  --out   docs/samples/muli4_seg6_review.json
# 打开 http://127.0.0.1:8765/ ，逐句核对，改动实时写盘
```

界面会**自动用 ⚑ 标出需重点核对的句子**（置信度 <0.85 / 被块级复核改过 / 说话人是未知·其他）。
密集多人场景可配合 `tools/mcp_review_server.py`（见 `docs/mcp_dense_review_guide.md`）。

**`review.json` 取值约定（固化工具据此解析，务必遵守）：**

| 情形 | 在复核里填的值 | 例 |
|---|---|---|
| 改成某具名角色 | 角色全名 | `甘织玲奈子` |
| 群众/群体（不计入准确率） | `群众·<群体名>` | `群众·摄影师们` |
| 两个说话人都说得通 | `主说话人/备选` | `长谷川同学/平野同学` |

> 复核值就是最终答案本身，**不要写成提示或缩写**。群众一律带 `群众·` 前缀；斜杠表示「都算对」，
> 主说话人取斜杠前第一个。

### Step 4 — 固化 → `groundtruth.json`（+ 重生成 transcript）

```bash
.venv/bin/python tools/build_groundtruth.py --seg muli4_seg6
```

把 review 的修正机械叠加到 parse 上，算出群众数/具名数/修正数/具名准确率，写出权威
`groundtruth.json`，并**重新生成与最终 parse 一致的 transcript.txt**。全程零手工计数。

### Step 5 — 验收（必须 0 错误才提交）

```bash
.venv/bin/python tools/verify_sample.py --seg muli4_seg6
```

交叉校验 parse / review / groundtruth 三者一致（段数、文本、修正落地、corrected 标记、
corrected_indices、以及 5 项口径自洽），**并校验 parse 确实是 BookVoiceParser 完整流水线产物**
（stats 含 `block_review`/`address_term_backcheck`/`scene_state` 阶段、`attribution_type` 为小写
JSON 值、段内含 `quote_id`/`candidates`/`scene_characters` 等字段）。输出 `✓ PASS` 才算合格。

> ⚠️ 常见翻车点：图省事用「裸 `agnes-2.0-flash` 单遍直出」代替 Step 2 的完整流水线。这种 parse
> 缺少复核阶段、字段残缺、`attribution_type` 会是 `"AttributionType.IMPLICIT"` 这种枚举 repr，
> **`model_speaker` 是未经纠错的弱基线，与 seg1-5 不可比**。`verify_sample.py` 现在会直接 FAIL 拦下。
> 已知一例：`muli4_seg6_bareflash_*` 即裸单遍版本（具名准确率 92.89%），作为对照基线保留，见下。

---

## 3. 口径定义（评测怎么算分）

- **群众（crowd，不计分）**：speaker 以 `群众·` / `厕所女生` 开头，或属于
  `{未知, 未知临时人物, 旁白, 其他, ""}`。
- `crowd_segments` = 终值说话人为群众的段数。
- `named_total` = `total_segments − crowd_segments`（计分的具名句总数）。
- `named_corrections` = 具名句里模型答错的数量（模型值不在 acceptable 集合内）。
- `model_named_accuracy` = `(named_total − named_corrections) / named_total`。

`groundtruth.json` 段结构：`{i, speaker, model_speaker, text, corrected[, acceptable]}`。
`speaker` 是人工终值，`model_speaker` 是模型原值，二者不同则 `corrected=true`；斜杠 acceptable-set
会额外带 `acceptable:[...]`，评测时模型命中其中任一即算对。

**场景难度分桶**（`tools/eval_external_model.py`）：以某句为中心 ±4 句窗口内的不同具名说话人数，
≤2 人记「简单」、≥3 人记「密集」。对照 `agnes-2.0-flash`：简单 ~96.7% / 密集 ~82% / 整体 ~93%。
seg5 是平静双人戏，准确率 97.89% 偏高属正常；密集多人段（如 seg4 的 cosplay 现场）会明显更低，
两类样本互补。

---

## 4. 评测某个外部模型

```bash
MODEL_BASE_URL=https://omnitok.xyz/v1 MODEL_NAME=gpt-5.5 MODEL_API_KEY=sk-... \
  .venv/bin/python tools/eval_external_model.py
```

在样本上跑任意 OpenAI 兼容模型的全流水线，按 groundtruth 打分并分难度桶输出。

---

## 5. 对照基线（reference_baseline）

如果想保留某种**非生产流水线**的解析做对比（如裸单遍模型），把它另存为 `*_bareflash_*` 一类的
独立文件集，并在其 `groundtruth.json` 顶层加 `"reference_baseline": true`（再补一句 `reference_note`
说明用途）。这样：

- `verify_sample.py` 仍校验它内部一致，但**豁免流水线溯源**（降级为提示，不算 FAIL）；
- 它不会被误当成权威样本参与流水线准确率统计。

现有一例：`muli4_seg6_bareflash_*` = seg6 切片上裸 `agnes-2.0-flash` 单遍直出 + 人工复核，具名
准确率 **92.89%**。留作和完整流水线版 `muli4_seg6_*` 在**同一段原文**上对比（看流水线复核到底加了
多少分；注意两者切分段数不同，需按文本而非段号对齐比较）。

## 6. 历史样本说明

`seg1`~`seg4` 早于本固化工具，用的是手工记账：早期把群众写成裸名（如 `女生小团体`，
groundtruth 里才补 `群众·` 前缀）、`named_corrections` 有 ±1~2 的人工漂移、seg2/seg3 没有口径字段。
因此 `verify_sample.py --all` 只有 **seg5 通过**，其余报「口径不符」属预期，不是 bug。它们的
groundtruth 段内容仍可用于评测；如需统一口径，可在保留其 acceptable-set / 群众标注后重新固化。
**新样本（seg6+）一律按本文 seg5 约定生产。**
