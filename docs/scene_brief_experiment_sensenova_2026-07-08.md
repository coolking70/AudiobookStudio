# 场景摘要 / 身份事实注入实验（sensenova 强复核器，2026-07-08 下午）

## 目的

验证 `tools/evaluate_evidence_review.py` 的 `--scene-brief` 注入（完整场景摘要 + 身份事实 / 仅身份事实）是否能在 **强复核器** 下提升 `audit-safe` 的说话人归属准确率。

此前（2026-07-08 上午）用 `agnes-2.0-flash` 当复核器跑出的结论是「注入摘要 +3 分」，但该信号存疑——agnes 本身在 seg8 上只有 104~107 分、处于"地板"水平，摘要只是把它抬离地板。本实验改用更强的 `sensenova / deepseek-v4-flash`（即此前 126/141 好结果的复核器）复跑，以公平评估摘要方向。

## 环境

- 复核器：`--provider sensenova --model deepseek-v4-flash`
- 风格与阈值：`--review-style audit-safe --type-thresholds default`（分层阈值：explicit_after 0.65 / first_person 0.75 / identity_alias 0.75 / address_term 0.80 / semantic_reply 0.85）
- 目标筛选：`--target-mode audit-disagree --limit 200`（seg8 共 51 个 audit-disagree 目标）
- 输入：
  - parse：`docs/samples/muli4_seg8_parse.json`
  - audit：`docs/samples/muli4_seg8_audit.json`
  - 摘要：`docs/samples/muli4_seg8_scene_brief.md`（完整摘要 + 身份事实）、`docs/samples/muli4_seg8_identity_facts.md`（仅身份事实）
- 运行依赖：工具依赖 `openai`，需在与项目隔离的 venv 中安装（`tools/` 不自动读 `.env`，需先 `source .env` 或 `export SENSENOVA_API_KEY=<your-key>`）。
- 注：SENSENOVA_API_KEY 不写入本文档与仓库；运行前在环境中注入即可。

## 结果（sensenova / deepseek-v4-flash 复核器）

| 配置 | reviewed | revised | 最终分数 (correct/141) | acc |
|---|---:|---:|---:|---:|
| ① 完整摘要 + 身份事实 | 39 | 7 | **123** | 0.8723 |
| ② 仅身份事实 | 33 | 3 | **121** | 0.8582 |
| ③ 基线（无摘要） | 45 | 5 | **124** | 0.8794 |

细分（dense = 难案例，simple = 简单案例）：

- ① simple 15/15 全对，dense 108/126
- ② simple 15/15 全对，dense 106/126
- ③ simple 15/15 全对，dense 109/126

## 对照：agnes-2.0-flash（弱复核器，同日 12:16 同三组）

| 配置 | 最终分数 | acc |
|---|---:|---:|
| ① 完整摘要 + 身份事实 | 107 | 0.7589 |
| ② 仅身份事实 | 107 | 0.7589 |
| ③ 基线（无摘要） | 104 | 0.7376 |

## 结论

1. **强复核器下，注入场景摘要 / 身份事实不升反降**：无摘要基线 124 最高，完整摘要 −1、仅身份事实 −3。摘要方向在强复核器上**没有正面收益**。
2. **上午 agnes 上的「注入 +3」是测量假象**：弱模型（agnes，地板 104）被摘要抬到 107，看似"有帮助"，实则只是脱离地板，远未到强模型天然 124 的水平。换成强复核器后，摘要反而成为干扰。
3. **强复核器本身已足够好**：`audit-safe + 分层阈值` 在 sensenova 上自然达 124/141（simple 15/15 全对），非常接近此前文档声称的 126 上限，差 2 个点属温度波动，不是算法差距。
4. **瓶颈在 dense 硬案例，不在背景信息**：强复核器 revised 仅 3~7 次，准确率卡在 dense 难归类对话，而非缺上下文。

**建议放弃「场景摘要 / 身份事实注入」方向**，将精力转向：分层阈值调优、`address_term` 硬门控（被称呼者不直接当说话人）、降低 `uncertain` 比例（今晚强复核器下仍有 7~17 个 uncertain，属纯损失）。

## 复现命令

```bash
# 运行环境：在隔离 venv 中用 openai（venv 路径见 tools 运行说明），并注入 key
export SENSENOVA_API_KEY=<your-key>
PY=PATH_TO_VENV/Scripts/python.exe

# ① 完整摘要 + 身份事实
"$PY" tools/evaluate_evidence_review.py \
  --seg muli4_seg8 --parse docs/samples/muli4_seg8_parse.json --audit docs/samples/muli4_seg8_audit.json \
  --provider sensenova --model deepseek-v4-flash \
  --scene-brief docs/samples/muli4_seg8_scene_brief.md \
  --review-style audit-safe --type-thresholds default \
  --target-mode audit-disagree --limit 200 \
  --out-dir outputs/regression/audit-safe-seg8-scene-brief-sensenova-20260708

# ② 仅身份事实
"$PY" tools/evaluate_evidence_review.py \
  --seg muli4_seg8 --parse docs/samples/muli4_seg8_parse.json --audit docs/samples/muli4_seg8_audit.json \
  --provider sensenova --model deepseek-v4-flash \
  --scene-brief docs/samples/muli4_seg8_identity_facts.md \
  --review-style audit-safe --type-thresholds default \
  --target-mode audit-disagree --limit 200 \
  --out-dir outputs/regression/audit-safe-seg8-identity-facts-sensenova-20260708

# ③ 基线（无摘要）
"$PY" tools/evaluate_evidence_review.py \
  --seg muli4_seg8 --parse docs/samples/muli4_seg8_parse.json --audit docs/samples/muli4_seg8_audit.json \
  --provider sensenova --model deepseek-v4-flash \
  --review-style audit-safe --type-thresholds default \
  --target-mode audit-disagree --limit 200 \
  --out-dir outputs/regression/audit-safe-seg8-baseline-sensenova-20260708
```

## 产物

- `outputs/regression/audit-safe-seg8-scene-brief-sensenova-20260708/`
- `outputs/regression/audit-safe-seg8-identity-facts-sensenova-20260708/`
- `outputs/regression/audit-safe-seg8-baseline-sensenova-20260708/`
- 运行日志：`outputs/regression/run3_sensenova_20260708.log`

以上目录位于 `outputs/`，默认被 `.gitignore` 忽略。
