# 2026-07-06 基线结构化证据与复核流程实验

## 背景

问题：若在基线阶段要求 LLM 输出更充分的判断理由，是否有利于后续分析和纠错？

结论：方向成立，但不应要求长篇自然语言理由。更适合生产的是短结构化证据：

```text
E=<证据标签>;R=<风险标签>;S=<原文线索>
```

其中 `E` 表示正向证据类型，`R` 表示风险类型，`S` 保留极短原文锚点。这样后续复核模型可以验证/反驳基线依据，机器审计也可以按标签统计错误分布。

## 流程改动

新增 `BatchConfig.evidence_mode`：

- `short`：默认旧模式，保持兼容。
- `structured`：要求 BatchLLM 初判的 `evidence` 使用结构化依据格式。

新增/更新工具：

- `tools/run_regression.py run --evidence-mode structured`
- `tools/parse_sample.py --evidence-mode structured`
- `tools/analyze_evidence_regression.py`：统计结构化依据覆盖率、证据/风险标签准确率、复核路由召回/精度。

## Agnes seg8 受控测试

样本：`muli4_seg8`，多人密集场景，141 个具名段。

为隔离“基线阶段依据”本身，本轮跑 `--no-block-review`。完整 block review 版本曾在 block review API 请求处长时间等待，未作为本轮有效结果。

| 配置 | 准确率 | 简单≤2人 | 密集≥3人 |
|---|---:|---:|---:|
| 旧短依据，no block review | 101/141 = 71.63% | 12/15 = 80.0% | 89/126 = 70.6% |
| 结构化依据，no block review | 100/141 = 70.92% | 12/15 = 80.0% | 88/126 = 69.8% |
| 已提交完整 Agnes 基线（含 block review） | 119/141 = 84.40% | 15/15 = 100.0% | 104/126 = 82.5% |

解读：

- 结构化依据相对旧短依据只低 1 段（-0.71%），没有出现明显“解释拖累初判”的证据。
- 与完整基线的差距主要来自关闭 block review，而不是结构化依据本身。
- 结构化依据会增加 prompt 复杂度，当前 Agnes 上游偶发慢请求/404，需要继续观察吞吐稳定性。

## 结构化依据统计

结构化版本：

- 结构化依据覆盖：137/141 = 97.2%
- `E=reply_turn`：18/18 = 100.0%
- `E=explicit_after`：17/19 = 89.5%
- `E=explicit_before`：26/38 = 68.4%
- `E=dialogue_alternation`：15/24 = 62.5%
- `E=narrator_anchor`：18/29 = 62.1%
- `R=multi_speaker`：1/5 = 20.0%

初步判断：

- `reply_turn`、`explicit_after` 是较强正证据。
- `dialogue_alternation`、`narrator_anchor` 在密集场景风险较高，应天然进入复核或降低自动覆盖权重。
- `explicit_before` 标签被 Agnes 用得偏宽，仍需让复核模型检查 `S=` 是否真是动作主语，而不是被观察者/受话人。
- 模型自己给出的 `R=` 风险标签偏保守不足；后处理的密集场景路由仍然必要。

## 建议流程

1. Agnes 保持默认基线，但在实验/复核包模式开启 `evidence_mode=structured`。
2. 机器审计先按结构化证据分层：
   - 高优先复核：`R` 非空、`confidence < 0.7`、`E` 包含 `dialogue_alternation/narrator_anchor/address_term/scene_presence`。
   - 低优先确认：`E=reply_turn` 或 `explicit_after` 且 `R=none`。
3. DeepSeek/SenseNova 复核 prompt 不再单纯重判 speaker，而是：
   - 判断基线 `E/R/S` 是否成立；
   - 给出是否推翻基线的更强原文证据；
   - 只在“基线证据不成立 + 新证据更强”时建议覆盖。
4. 自动覆盖仍不建议全开；应先做多模型一致性或人工确认门控。

## 下一步

- 在 `Agnes -> SenseNova deepseek-v4-flash` 复核脚本里加入“验证/反驳基线结构化证据”的 prompt。
- 跑完整 `seg8` structured + block review；若 block review 继续慢，先复用已落盘的 no-block parse 做 DeepSeek 复核验证。
- 调整 Agnes 结构化 prompt：强调 `explicit_before` 必须是动作主语，不能把紧前文被提及对象当作正证据。

## TokenHub 复核烟测

本机当前可用 API：

- `AGNES_API_KEY`：可用。
- `TOKENHUB_API_KEY`：可用。
- `SENSENOVA_API_KEY`：本轮由用户临时提供，未写入仓库或文档。
- 通用 `MODEL_BASE_URL`：未配置。

因此先用 TokenHub 做流程烟测。`deepseek-v4-flash` 被 TokenHub 拒绝：

```text
model deepseek-v4-flash not in allowed list
```

改用旧评测中可用的 `deepseek-v3.1-terminus`，对 `muli4_seg8` structured no-block parse 的前 24 个 flagged 目标进行“验证/反驳基线证据”复核。

| 设置 | 复核条数 | 自动改判 | keep | uncertain | 分数 |
|---|---:|---:|---:|---:|---:|
| 覆盖阈值 0.75 | 24 | 5 | 15 | 4 | 100/141 = 70.92% |
| 覆盖阈值 0.85 | 24 | 1 | 17 | 6 | 101/141 = 71.63% |

0.75 阈值能修对部分错段，但也误伤正确段；0.85 阈值只自动覆盖一条：

- `#16`：`琴纱月 -> 小柳香穗`，修正成功。

结论：

- “验证/反驳证据”的复核形式可跑通，且能产生有效修正。
- 自动覆盖必须高阈值；低阈值仍会被多人对话相位误判误伤。
- TokenHub `deepseek-v3.1-terminus` 可用于流程验证，但不能代表文档中最佳的 SenseNova `deepseek-v4-flash`。

## SenseNova deepseek-v4-flash 复核烟测

用户临时提供 SenseNova key 与 endpoint 后，`deepseek-v4-flash` 可正常调用。

### flagged 前 24 条

目标：structured no-block parse 的前 24 个 flagged 目标，覆盖阈值 0.85。

| 复核条数 | 自动改判 | keep | uncertain | 分数变化 |
|---:|---:|---:|---:|---:|
| 24 | 4 | 19 | 1 | 100/141 = 70.92% |

四个自动改判中：

- 修对：`#8 甘织玲奈子 -> 小柳香穗`、`#9 帕曼小姐 -> 甘织玲奈子`
- 误伤：`#7 甘织玲奈子 -> 小柳香穗`、`#24 琴纱月 -> 甘织玲奈子`

结论：SenseNova 能修错，但“flagged 目标 + 单模型高置信”仍不足以自动覆盖，尤其容易在密集多人连续话轮里把相邻句相位看错。

### audit-disagree 门控

目标发现策略：只挑机器审计第二意见 `reask` 与 Agnes 当前 `speaker` 分歧，且该段有低置信/风险或弱证据标签。

离线看目标集：

| 目标策略 | 目标数 | 真实错段 | 错段占比 |
|---|---:|---:|---:|
| flagged | 135 | 41 | 30.4% |
| audit-disagree | 66 | 33 | 50.0% |

SenseNova `deepseek-v4-flash` 跑 `audit-disagree` 目标后：

| 复核目标 | 返回 | 自动改判 | 修对 | 误伤 | 仍错改判 | 净收益 | 分数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 66 | 60 | 24 | 13 | 3 | 8 | +10 | 110/141 = 78.01% |

进一步增加“强反证 reason gate”：只自动应用 reason 中含明确证据词的改判，例如“前文明示/后文明示/明确/证实/第一人称/屏幕/被拉/语气喵”等。

| 门控 | 自动改判 | 修对 | 误伤 | 仍错改判 | 净收益 | 分数 |
|---|---:|---:|---:|---:|---:|---:|
| audit-disagree + strong reason | 10 | 9 | 0 | 1 | +9 | 109/141 = 77.30% |

这比裸 `audit-disagree` 少 1 个净收益，但消除了本轮观察到的自动误伤，更适合作为产品默认自动覆盖策略；其余 SenseNova 建议保留为“待人工确认”。

### oracle 错段 + 正确对照诊断

目标：全部已知错段 + 正确对照，60 条；覆盖阈值 0.85。该设置使用真值挑目标，只用于测复核能力上限，不是生产口径。

| 复核条数 | 自动改判 | 修对 | 误伤 | 仍错改判 | 净收益 | 分数 |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 18 | 15 | 0 | 3 | +15 | 115/141 = 81.56% |

结论：

- SenseNova `deepseek-v4-flash` 对“真正错段”的证据反驳能力很强。
- 当前瓶颈转为目标发现/门控，而不是复核模型本身。
- 下一步应把自动目标从“所有 low confidence / dense flagged”收紧为多信号交集，例如：
  - `E` 为 `dialogue_alternation/narrator_anchor/scene_presence`
  - 且机器审计/第二意见与基线分歧
  - 且 SenseNova 给出 `confidence >= 0.85`
  - 且反证 reason 含明确动作主语/后文说话标记/角色身份语气，而不是泛泛“后文继续说话”

落地脚本：

```bash
SENSENOVA_API_KEY=... .venv/bin/python tools/evaluate_evidence_review.py \
  --seg muli4_seg8 \
  --parse outputs/regression/evidence-structured-seg8-noblock-20260706/muli4_seg8_parse.json \
  --audit docs/samples/muli4_seg8_audit.json \
  --provider sensenova \
  --model deepseek-v4-flash \
  --target-mode audit-disagree \
  --min-confidence 0.85 \
  --reason-gate strong
```

## 完整 Agnes 基线还原测试

用户要求测试是否能还原旧结论中 `Agnes + SenseNova deepseek-v4-flash = 97.87%` 的效果。本轮改用已提交的完整 Agnes 基线：

- 输入 parse：`docs/samples/muli4_seg8_parse.json`
- 基线分数：119/141 = 84.40%
- 模型：SenseNova `deepseek-v4-flash`

### oracle 错段 + 正确对照

目标：22 个已知错段 + 22 个正确对照。该口径用于验证复核能力上限，但使用真值挑目标，不是生产流程。

| 设置 | reviewed | 自动改判 | 修对 | 误伤 | 仍错改判 | 净收益 | 分数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| confidence >= 0.85 | 44 | 13 | 8 | 2 | 3 | +6 | 125/141 = 88.65% |
| confidence >= 0.70 | 38 | 17 | 9 | 6 | 2 | +3 | 122/141 = 86.52% |

结论：当前 `evaluate_evidence_review.py` 的“验证/反驳结构化证据”prompt 没有还原旧文档的 97.87% 结果。降低阈值不会接近 97%，反而增加误伤。

主要差异：

- 旧 97.87% 来自当时的交叉复核矩阵口径，记录为 Agnes 错段 `19/22` 修对、正确对照 `0/12` 误报。
- 当前脚本给复核模型的是“验证/反驳基线证据”的任务，并且完整 Agnes parse 的 `evidence` 不是本轮 structured `E/R/S` 初判证据。
- moon/帕曼/纱月等别名/临时身份在当前 prompt 中仍会混淆，导致若干错改。
- 当前测试说明：SenseNova 仍有纠错能力，但此脚本不是旧交叉矩阵的等价复现。

### 自动门控

目标：`audit-disagree + strong reason gate`，即只挑机器审计第二意见与当前 speaker 分歧的段，并且只自动覆盖有强反证 reason 的建议。

| 目标 | reviewed | 自动改判 | 修对 | 误伤 | 净收益 | 分数 |
|---:|---:|---:|---:|---:|---:|---:|
| 51 | 45 | 1 | 1 | 0 | +1 | 120/141 = 85.11% |

结论：完整 Agnes 基线已经较强，当前自动门控非常保守，只能稳定捞回 1 条。本轮未能端到端复现 97.87%；要复现旧结论，需要恢复旧交叉矩阵复核 prompt/目标构造，或找回当时的 `outputs/regression/cross_review_muli4_seg8_with_glm52.json` 原始产物。
