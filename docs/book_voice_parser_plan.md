# BookVoiceParser 开发规划

> 生成日期：2026-04-28  
> 参考文档：`../BookVoiceParser/小说角色对话分析与中文BookNLP替代方案可行性报告.docx`

---

## 一、背景与目标

### 现状痛点

AudiobookStudio 当前路径：

```
原文 → LLM 一次性产出 Segment[] → 角色匹配音色 → TTS 拼接
```

该路径完全依赖 LLM 自由生成，存在以下问题：

| 问题 | 表现 |
|---|---|
| 格式不稳定 | JSON 结构偶发错误，需要大量兼容代码 |
| 隐式说话人误判 | 连续对话无"某某说"时容易串角 |
| 长上下文衰减 | 章节末尾角色归属质量下滑 |
| 角色名漂移 | 同一角色被模型用不同别名输出 |
| 成本高 | 全量文本每次走 LLM，token 消耗不可控 |

### 目标

构建 **Chinese-BookNLP-Lite**：一个可监督、可评测、可持续改进的中文小说处理子模块，将 LLM 降级为"低置信度复核层"。

最终暴露统一入口，兼容现有 narrate 流水线：

```python
from book_voice_parser import parse_novel

segments = parse_novel(text, role_hints=None, llm_config=None)
# 返回值兼容 schemas.Segment，可直接送入 pipeline.py
```

---

## 二、架构设计

### 模块结构

```
book_voice_parser/
├── __init__.py             # 对外接口：parse_novel()
├── schema.py               # SegmentEx = Segment + addressee/confidence/evidence/attribution_type
├── cleaner.py              # 引号统一、章节切分、OCR 清洗
├── quote_extractor.py      # 台词抽取（中文引号、冒号、连续对话）
├── candidate_gen.py        # 角色候选生成（HanLP NER + 动作主体 + 称谓 + 最近活跃角色）
├── rule_attributor.py      # 显式归属规则（"X说道/问/笑道"等正则）
├── implicit_attributor.py  # 隐式说话人模型（SPC 风格）
├── consistency_fixer.py    # 全局一致性修正（轮替检测、场景约束、跳变检查）
├── review_router.py        # 低置信度复核路由（批量送 LLM）
├── alias_registry.py       # 别名表（复用/扩展 character_registry.py）
└── eval/
    ├── gold_zh.jsonl       # 人工金标评测集
    ├── run_eval.py         # 评测脚本
    └── baseline_results/   # 各版本基线结果存档
```

### 对接点

| 现有文件 | 对接方式 |
|---|---|
| `schemas.py` | `Segment` 增加 `addressee / confidence / evidence / attribution_type` 可选字段 |
| `character_registry.py` | `alias_registry.py` 直接复用，扩展别名归并能力 |
| `llm_client.py` | `review_router.py` 调用，仅处理低置信度片段 |
| `app.py` | 新增 `/api/parse_v2` 端点，旧 `/api/parse` 保留至 v1.0 |
| `pipeline.py` | 无需改动，narrate 流程接收兼容的 `Segment[]` 即可 |

### 数据流

```
小说文本
  │
  ▼
① cleaner.py          → 清洗后文本
  │
  ▼
② quote_extractor.py  → [(quote_text, context_before, context_after), ...]
  │
  ▼
③ candidate_gen.py    → 每条台词附加 candidates[], scene_characters[]
  │
  ▼
④ rule_attributor.py  → 显式归属（高置信度直接输出，跳过 ⑤）
  │
  ▼
⑤ implicit_attributor.py  → (speaker, confidence)
  │
  ▼
⑥ consistency_fixer.py    → 轮替修正、跳变标记
  │
  ▼
⑦ review_router.py        → confidence < 阈值 → LLM 复核
  │
  ▼
⑧ SegmentEx[]             → 送入现有 narrate 流水线
```

### 扩展后的 Segment 字段

```python
class SegmentEx(Segment):
    addressee: Optional[str] = None          # 受话人
    confidence: float = 1.0                  # 归属置信度 0–1
    evidence: Optional[str] = None           # 归属依据（供人工复核）
    attribution_type: Optional[str] = None   # explicit / implicit / latent / group / narrator
```

---

## 三、分阶段路线图

### 阶段 0｜基线评测集（1 周）

**目标**：建立量化基准，所有后续"提升"都依赖此集合验证。没有评测集就没有证据。

**任务**：
- 从 `samples/` 现有章节 + 1–2 本网络小说各取代表段落
- 人工标注 2,000–5,000 条台词，覆盖显式/隐式/多人同场三类
- 标注格式：

```jsonl
{
  "quote_id": "novel001_ch003_q015",
  "quote_text": "你别过来。",
  "speaker": "陆沉",
  "addressee": "苏晚",
  "evidence": "陆沉后退一步，低声道",
  "attribution_type": "explicit_after",
  "scene_characters": ["陆沉", "苏晚", "黑衣人"],
  "candidates": ["陆沉", "苏晚", "黑衣人", "旁白", "未知"],
  "confidence": 1.0
}
```

- 用现有 LLM 路线跑一次 baseline，按以下指标记录基准值：

| 指标 | 含义 |
|---|---|
| Quote Extraction F1 | 台词边界准确率 |
| Speaker Accuracy（总） | 说话人整体准确率 |
| Explicit Accuracy | 显式说话人准确率（应接近规则上限） |
| Implicit Accuracy | 隐式说话人准确率（核心提升目标） |
| Multi-character Accuracy | 多人同场准确率 |
| Unknown/Uncertain Rate | 模型主动标记不确定的比例 |

**退出条件**：金标集 ≥ 2,000 条，baseline 数值入档。

---

### 阶段 1｜规则与候选层 v0.1（2 周）

**目标**：用规则吃掉显式归属，用 HanLP 生成候选列表。

**任务**：
- `cleaner.py`：统一 `「」""''` 引号，修正章节边界，过滤内心独白/旁白标记
- `quote_extractor.py`：支持中文冒号 + 引号、连续对话（无明确说话人的后续引号）
- `candidate_gen.py`：引入 `hanlp`（CPU），抽取 NER 人名、动作主体、称谓（师姐/殿下/他/她）、近 N 句最近出现的角色
- `rule_attributor.py`：正则覆盖 `X说道/X问/X笑道/众人齐声道/X心想` 等模式；`众人/群体` 归为 `group`；内心独白单独标记
- `alias_registry.py`：复用 `character_registry.py`，增加"候选归一化"能力（别名 → canonical 名）

**依赖**：`hanlp` 加入 `requirements.txt`

**退出条件**：
- 显式归属准确率 ≥ 95%
- 候选生成召回率（正确说话人在候选列表中）≥ 90%

---

### 阶段 2｜隐式说话人模型 v0.2（2–4 周）

**目标**：解决无明显说话人标记的连续对话归属。

**策略**：先用本地 LLM（`local_llm.py`）做 SPC zero-shot 模拟，验证方向收益后再考虑微调。

**SPC 思路**（优先）：
- 将候选角色符号化（A、B、C…）
- 构造 prompt：`上下文 + 台词 + "以下哪个角色说了这句话？[A=陆沉, B=苏晚, C=旁白]"`
- 输出候选编号 + 置信度

**CSI 思路**（对比备选）：
- MRC / span extraction：让模型从上下文中抽取说话人片段
- 适合候选列表不完整的场景

**退出条件**：
- 隐式准确率较 baseline 提升 ≥ 5 个点，或在持平准确率下成本下降 ≥ 50%
- 多人同场场景准确率有明显改善

---

### 阶段 3｜全局一致性 + 复核层 v0.3（1–2 周）

**目标**：修正跨句/跨段的归属漂移，把疑难片段精准投喂给 LLM。

**任务**：
- `consistency_fixer.py`：
  - 连续对话轮替检测（ABAB 模式推断）
  - 场景人物约束（台词说话人必须在 `scene_characters` 中）
  - 异常跳变检查（同一句前后说话人与上下文完全无关）
  - 对修正后样本打 `confidence` 折扣
- `review_router.py`：
  - `confidence < threshold`（默认 0.7）的片段批量打包
  - 调用 `llm_client.py`，复用现有 LLM 配置
  - 结果写入 `review.jsonl`，可回流修正或用于后续训练
- 输出 `addressee` 字段，为情绪/风格判断提供"谁在说、对谁说"的上下文

---

### 阶段 4｜AudiobookStudio 集成（1 周）

**任务**：
1. **Schema 扩展**：`schemas.py` 的 `Segment` 增加 `addressee / confidence / evidence / attribution_type` 四个可选字段，旧调用零影响
2. **API 层**：`app.py` 新增 `/api/parse_v2`，请求体复用 `ParseRequest`，响应体兼容现有前端
3. **前端开关**：解析面板加单选「LLM 直解（旧）/ 结构化解析（新）」，默认走旧路线，灰度切换
4. **角色衔接**：解析器输出的 canonical 名称直接作为 `RoleProfile` 的 key，自动复用声音库匹配逻辑
5. **低置信度可视化**：`confidence < 0.7` 的段落在前端高亮，点击展开 `evidence` 供人工微调

---

### 阶段 5｜主动学习闭环（持续迭代）

- 用户在前端的人工修正自动落盘到 `book_voice_parser/data/silver/`
- 周期性将银标数据回灌 SPC 模型微调
- 用 GPT/Claude 蒸馏处理最疑难的低置信度样本，加入金标集
- 定期跑 `eval/run_eval.py` 对比各版本指标，入档 `eval/baseline_results/`

---

## 四、关键决策

| 决策点 | 建议 | 理由 |
|---|---|---|
| 是否新建独立仓库 | **否**，作为子包放在本仓库内 | 共用 `character_registry`、`llm_client`、`schemas` |
| 中文 NLP 组件选型 | **HanLP 优先**，LTP 备选 | 覆盖最广（分词/NER/依存/SRL/共指） |
| 是否立即训模型 | **否**，先用 SPC zero-shot + 规则跑通 | 无评测集时训练无法判断真实收益 |
| LLM 角色定位 | 降级为**复核层 + 教师模型** | 解决格式不稳、长上下文衰减、高成本问题 |
| 旧接口兼容性 | 旧 `/api/parse` **保留**至 v1.0 | 灰度迁移，防回归 |
| 是否让模型自由生成角色名 | **否**，必须提供候选列表 | 消除幻觉和角色名漂移 |

---

## 五、风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| 公开数据风格偏窄 | WP/JY 不覆盖轻小说/网文 | 建立自有金标集，按目标书型分层采样 |
| 角色别名归并困难 | 同一角色有姓名/称谓/绰号/身份名 | 建角色知识库；主角/高频角色先人工维护；低频允许 unknown |
| 隐式对话准确率有天花板 | 连续多人对话无法 100% 自动判断 | 设置置信度阈值；允许输出"不确定"，优于自信地错 |
| OCR 错误污染训练 | 引号/角色名/标点错误导致标注错 | 训练前做清洗；OCR 错误单独分类，不混入模型错误统计 |
| 模型指标与听感不一致 | 文本准确率高但 TTS 仍串角 | 增加人工听感抽检；设立"TTS Impact Error"指标 |
| LLM 复核成本失控 | 低置信度比例过高 | 先优化规则层降低模糊片段比例；只送真正冲突/多人样本 |

---

## 六、最小可运行 POC（验收标准）

在正式进入各阶段开发前，用以下最小版本验证整体路线可行：

| 模块 | POC 要求 |
|---|---|
| 输入 | `samples/` 中任意 1–2 章，人工修正明显引号/OCR 错误 |
| 台词抽取 | 识别中文引号内台词，保留上下文前后各 3–8 句 |
| 角色表 | 手工提供主要角色与别名表（JSON 格式） |
| 规则层 | 处理"某某说/问/道/喊/笑道"等显式归属 |
| 模型层 | 本地 LLM zero-shot SPC 格式，不急于训练 |
| 复核层 | 低置信度样本输出 `review.jsonl`，供人工或高价模型校对 |
| 输出 | `SegmentEx[]` JSON：speaker/text/style/emotion/confidence/evidence |

---

## 七、参考资料

| 编号 | 资料 | 用途 |
|---|---|---|
| [1] | [BookNLP](https://github.com/booknlp/booknlp) | 架构参考 |
| [2] | [WP 中文数据集](https://www.isca-archive.org/interspeech_2019/chen19d_interspeech.html) | 评测基准参考 |
| [3] | [WP GitHub](https://github.com/YueChenkkk/Chinese-Dataset-Speaker-Identification) | 标注格式参考 |
| [4] | [CSI NAACL 2022](https://aclanthology.org/2022.naacl-main.165/) | 端到端路线对比 |
| [5] | [CSI GitHub](https://github.com/yudiandoris/csi) | MRC 路线实现 |
| [6] | [SPC EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.225/) | 隐式归属核心方案 |
| [7] | [SPC GitHub](https://github.com/YueChenkkk/SPC-Novel-Speaker-Identification) | 模型层优先复现对象 |
| [8] | [JY-QuotePlus](https://arxiv.org/html/2408.09452v1) | addressee 字段扩展参考 |
| [9] | [HanLP](https://github.com/hankcs/hanlp) | 候选生成 NLP 组件 |
| [10] | [LTP](https://github.com/HIT-SCIR/ltp) | 备选 NLP 组件 |
