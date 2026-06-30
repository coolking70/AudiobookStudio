# 2026-07-01 多 API 模型主流水线与复核交叉评测

本文汇总 2026-06-30 至 2026-07-01 对 LongCat、SenseNova、阿里云 MaaS 若干模型的补充评测。评测目标是判断它们适合替代 Agnes 主流水线，还是更适合接入复核/审计阶段。

所有测试均不提交 API key。原始运行产物保存在 `outputs/regression/`，该目录为本地运行产物，不进入版本库。

## 评测口径

- 主样本：`muli4_seg8`，多人密集场景，Agnes 基线最弱，适合暴露模型上限。
- 主流水线测试：用 `tools/eval_external_model.py` 跑完整 BatchLLM 流水线，再对照 `muli4_seg8_groundtruth.json` 评分。
- 复核测试：固定某个基线 parse，对已知错段和正确对照段做聚焦重问，统计错段纠对率、正确段误报率。
- 综合准确率：在 `muli4_seg8` 的 141 个具名段上，用“基线正确数 + 复核自动覆盖净收益”折算。

## 主流水线结果

| 模型/来源 | `muli4_seg8` 准确率 | 密集桶 | 备注 |
|---|---:|---:|---|
| SenseNova `glm-5.2` | **92.91%** | 116/126 | 单样本最高；吞吐较慢，reasoning 开销高 |
| LongCat-2.0 | 92.20% | 116/126 | 密集场景强；简单样本上有回退和内容审核风险 |
| 阿里云 `deepseek-v4-flash` | 91.49% | 114/126 | 均衡，适合候选复核/难段模型 |
| 阿里云 `qwen3.7-max-preview` | 91.49% | 115/126 | 密集桶强但较慢 |
| 阿里云 `qwen3.6-35b-a3b` | 88.65% | 110/126 | 复核阶段较强，但完整流水线一般 |
| 阿里云 `kimi-k2.6` | 88.65% | 111/126 | 快，完整流水线一般 |
| SenseNova `deepseek-v4-flash` | 88.65% | 110/126 | 可用但不如同源 `glm-5.2` |
| Agnes 基线 | 84.40% | 约 82.5% | 当前默认主流水线 |

结论：强模型直接替换主流水线可以显著提升 `seg8`，但没有一个模型在成本、吞吐、稳定性和普通样本表现上全面压过 Agnes。生产上更稳的策略仍是 Agnes 做默认基线，强模型做异构复核。

## 复核阶段结果

### LongCat / SenseNova 复核

在 `muli4_seg8` 上，对 Agnes 错段和正确对照段做聚焦复核：

| 复核模型 | 错段检出 | 错段纠对 | 正确段误报 | 备注 |
|---|---:|---:|---:|---|
| SenseNova `deepseek-v4-flash` | 20/22 = 90.9% | **19/22 = 86.4%** | **0/12 = 0%** | 当前最佳 Agnes 异构复核器 |
| SenseNova `glm-5.2` | 未单独列入同批全量，但交叉矩阵为 17/22 | 17/22 = 77.3% | **0/12 = 0%** | 更保守，较慢 |
| LongCat-2.0 | 18/22 = 81.8% | 16/22 = 72.7% | 1/12 = 8.3% | 速度较好，可作快速第二意见 |

### 阿里云 Top 4 复核

完整 `muli4_seg8` 错段集：22 个错段 + 16 个正确对照段。

| 阿里云模型 | 错段纠对 | 正确段误报 | 耗时 | 备注 |
|---|---:|---:|---:|---|
| `qwen3.7-max-preview` | **19/22 = 86.4%** | 1/16 = 6.2% | 431.9s | 质量最高但慢 |
| `deepseek-v4-flash` | 17/22 = 77.3% | 2/16 = 12.5% | 91.1s | 较均衡 |
| `qwen3.6-35b-a3b` | 17/22 = 77.3% | 2/16 = 12.5% | 221.9s | 稳但慢 |
| `kimi-k2.6` | 17/22 = 77.3% | **0/16 = 0%** | **62.3s** | 快速低误报复核候选 |

阿里云 22 模型轻量初筛中，多个模型不可用或未开通：`kimi/kimi-*`、`ZHIPU/GLM-*` 等返回产品未激活；部分 Qwen 模型超时或返回不完整。实际值得继续评估的是上表四个模型，以及轻量初筛表现较好的 `qwen3.6-flash*`，但后者在加样本后出现误报，不建议自动覆盖。

## 交叉复核矩阵

补测 `glm-5.2` token 恢复后的交叉组合：

| 基线 -> 复核 | 错段纠对 | 正确段误报 | 自动覆盖净收益 |
|---|---:|---:|---:|
| Agnes -> SenseNova `deepseek-v4-flash` | **19/22 = 86.4%** | **0/12 = 0%** | **+19** |
| Agnes -> 阿里云 `qwen3.7-max-preview` | 19/22 = 86.4% | 1/16 = 6.2% | +18 |
| Agnes -> SenseNova `glm-5.2` | 17/22 = 77.3% | **0/12 = 0%** | +17 |
| Agnes -> 阿里云 `kimi-k2.6` | 17/22 = 77.3% | **0/16 = 0%** | +17 |
| LongCat -> SenseNova `glm-5.2` | 6/11 = 54.5% | **0/12 = 0%** | +6 |
| SenseNova `deepseek-v4-flash` -> `glm-5.2` | 9/16 = 56.2% | **0/12 = 0%** | +9 |
| SenseNova `glm-5.2` -> Agnes | 6/10 = 60.0% | 3/12 = 25.0% | +3 |
| SenseNova `glm-5.2` -> LongCat | 4/10 = 40.0% | 2/12 = 16.7% | +2 |
| SenseNova `glm-5.2` -> SenseNova `deepseek-v4-flash` | 3/10 = 30.0% | 1/12 = 8.3% | +2 |

关键发现：强复核模型并非对所有基线都安全。Agnes 作为基线时，异构模型的复核收益最大、误报最低；若基线已换成 LongCat 或 `glm-5.2`，再用 Agnes 或其它模型复核，误报会明显上升，收益下降。

## 综合准确率

按 `muli4_seg8` 141 个具名段折算“基线 + 复核自动覆盖”：

| 组合 | 综合准确率 |
|---|---:|
| **Agnes -> SenseNova `deepseek-v4-flash`** | **138/141 = 97.87%** |
| Agnes -> 阿里云 `qwen3.7-max-preview` | 137/141 = 97.16% |
| LongCat -> SenseNova `glm-5.2` | 136/141 = 96.45% |
| Agnes -> SenseNova `glm-5.2` | 136/141 = 96.45% |
| Agnes -> 阿里云 `kimi-k2.6` | 136/141 = 96.45% |
| Agnes -> LongCat | 134/141 = 95.04% |
| Agnes -> 阿里云 `deepseek-v4-flash` | 134/141 = 95.04% |
| SenseNova `glm-5.2` 单基线 | 131/141 = 92.91% |
| LongCat 单基线 | 130/141 = 92.20% |

当前最佳组合是 **Agnes 基线 + SenseNova `deepseek-v4-flash` 复核**。

## 接入建议

1. 保留 Agnes 作为默认主流水线基线。
2. 将 SenseNova `deepseek-v4-flash` 作为首选异构复核模型，优先用于 `machine audit` / 离线复核包 / 密集场景重判。
3. SenseNova `glm-5.2` 可作为保守低误报复核器，但不适合全量扫，主要受吞吐影响。
4. 阿里云 `qwen3.7-max-preview` 可用于少量高难段的高质量复核。
5. 阿里云 `kimi-k2.6` 可作为快速低误报第二意见，但纠对率略低。
6. 不建议直接把复核建议无人工确认地全量覆盖；即使最强模型也存在少量共同误伤。产品层应显示为“强模型建议”，由用户确认或至少通过多模型一致性门控后再应用。

## 产物索引

本轮主要结果文件：

- `outputs/regression/longcat_muli4_dense_probe_summary.json`
- `outputs/regression/sensenova_glm52_muli4_seg8_summary.json`
- `outputs/regression/sensenova_deepseek_v4_flash_muli4_seg8_summary.json`
- `outputs/regression/cross_review_muli4_seg8_no_glm52.json`
- `outputs/regression/cross_review_muli4_seg8_with_glm52.json`
- `outputs/regression/aliyun_22_review_probe_muli4_seg8_light.json`
- `outputs/regression/aliyun_top6_review_probe_muli4_seg8.json`
- `outputs/regression/aliyun_top4_review_full_muli4_seg8.json`
- `outputs/regression/aliyun_deepseek_v4_flash_muli4_seg8_summary.json`
- `outputs/regression/aliyun_qwen37_max_preview_muli4_seg8_summary.json`
- `outputs/regression/aliyun_qwen36_35b_a3b_muli4_seg8_summary.json`
- `outputs/regression/aliyun_kimi_k26_muli4_seg8_summary.json`
