# 2026-07-06 审计型安全复核实验

## 背景

旧的直接重判复核在 oracle 目标集上最终准确率较高，但本质上更接近“用强模型局部替代弱模型”。这能测出强模型上限，却不完全符合主流程设计理念：复核阶段应验证基线证据、发现风险，并只在证据足够强时自动覆盖。

本轮新增 `tools/evaluate_evidence_review.py --review-style audit-safe`，将强模型输出改为证据审计：

- `baseline_evidence_valid`：当前证据是否支持基线 speaker。
- `counter_evidence_type`：推翻基线的强反证类型。
- `independent_signal`：是否存在独立风险/分歧信号。
- `auto_apply_safe`：模型认为是否可安全自动覆盖。

自动覆盖门控：

1. `decision=revise`
2. `confidence >= 0.85`
3. `baseline_evidence_valid=false`
4. `counter_evidence_type` 属于强反证类型
5. `independent_signal=true`
6. `auto_apply_safe=true`

否则只写入待人工证据，不自动覆盖。

## 测试口径

样本：`muli4_seg1 / seg4 / seg5 / seg9`。

基线：已提交 Agnes 完整 parse。

目标集：oracle 诊断口径，即全部已知错段 + 最多 12 条正确对照。该口径用于比较复核能力和误伤风险，不是生产目标发现策略。

复核模型：SenseNova `deepseek-v4-flash`。

## 结果

| 方法 | 基线 | 复核后 | 修对 | 正确对照误伤 | 错段错改 |
|---|---:|---:|---:|---:|---:|
| 旧直接重判 | 838/910 = 92.09% | 866/910 = 95.16% | 33 | 5 | 18 |
| 原证据验证 | 838/910 = 92.09% | 859/910 = 94.40% | 26 | 5 | 6 |
| 审计安全门控 | 838/910 = 92.09% | 864/910 = 94.95% | 26 | 0 | 未单独统计 |

分样本：

| 样本 | 基线 | 旧直接重判 | 原证据验证 | 审计安全门控 |
|---|---:|---:|---:|---:|
| `muli4_seg1` | 186/211 | 197/211 | 195/211 | **199/211** |
| `muli4_seg4` | 241/260 | 242/260 | 241/260 | 241/260 |
| `muli4_seg5` | 276/282 | 279/282 | 277/282 | 277/282 |
| `muli4_seg9` | 135/157 | **148/157** | 146/157 | 147/157 |

## 解读

- 旧直接重判仍是最高分，但误伤正确对照 5 条，且错段错改较多，说明它确实有“强模型接管”倾向。
- 原证据验证更保守，但收益低于旧流程，且仍有正确对照误伤。
- 审计安全门控的最终准确率接近旧流程，同时本轮正确对照误伤为 0，更符合生产默认逻辑。
- `seg1` 上审计安全门控反而超过旧直接重判，说明“强反证 + 独立信号”并不必然牺牲准确率。
- `seg4` 上审计安全门控选择不自动改判，避免了旧流程在该样本上的误伤和错改。

## 建议

1. 保留旧直接重判作为“强模型替代上限评测”，不作为默认自动覆盖策略。
2. 将 `audit-safe` 作为下一轮产品化候选：自动覆盖只接受强反证；其余建议进入复核包。
3. 下一步需要补充生产目标发现口径，不再使用 oracle 错段：
   - 低置信或风险标签；
   - 机器审计第二意见分歧；
   - 块复核/主判不一致；
   - 密集多人场景与称呼/别名冲突。
4. 对 `audit-safe` 增加错段错改统计，区分“没有修到”和“修成另一个错 speaker”。

## 2026-07-07 分层阈值优化

`muli4_seg8` 首轮 `audit-safe` 使用统一 `confidence >= 0.85`，结果为 124/141，修对 5 条、误伤 0 条。复查发现模型对部分低置信但证据类型较强的建议判断正确，例如：

- `explicit_after`：前后文明确指出台词属于 moon/琴纱月。
- `first_person`：第一人称视角锚定明确。

因此新增 `--type-thresholds default`，只对部分反证类型降低阈值：

| counter_evidence_type | 阈值 |
|---|---:|
| `explicit_after` | 0.65 |
| `first_person` | 0.75 |
| `identity_alias` | 0.75 |
| `address_term` | 0.80 |
| `semantic_reply` | 0.85 |

注意：`explicit_before` 暂不降阈值。测试中模型会把“前文我发现/我内心后”这类弱线索也标成 `explicit_before`，若降到 0.65 会在 `seg8` 误伤正确对照。

复用已生成的审计输出重新应用默认分层阈值：

| 样本 | 统一 0.85 | 分层阈值 | 修对 | 误伤 |
|---|---:|---:|---:|---:|
| `muli4_seg1` | 199/211 | 199/211 | 13 | 0 |
| `muli4_seg4` | 241/260 | 242/260 | 1 | 0 |
| `muli4_seg5` | 277/282 | 277/282 | 1 | 0 |
| `muli4_seg8` | 124/141 | 126/141 | 7 | 0 |
| `muli4_seg9` | 147/157 | 147/157 | 12 | 0 |

五样本合计从 988/1051 提升到 991/1051，仍保持 0 条正确对照误伤。

结论：分层阈值比统一阈值更适合作为 `audit-safe` 默认策略，但必须保持 `explicit_before` 保守，后续若要放宽，应要求 reason 中出现真正动作主语证据，而不是普通前文叙述。

## 产物

- `outputs/regression/old_vs_evidence_review_multi_sample.json`
- `outputs/regression/audit-safe-seg1-20260706/`
- `outputs/regression/audit-safe-seg4-20260706/`
- `outputs/regression/audit-safe-seg5-20260706/`
- `outputs/regression/audit-safe-seg9-20260706/`
- `outputs/regression/audit-safe-seg8-20260706/`
- `outputs/regression/audit-safe-seg8-typed-thresholds-v2-reapply-20260707/`
