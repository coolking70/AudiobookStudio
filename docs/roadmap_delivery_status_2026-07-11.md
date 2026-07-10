# 路线图实施与验收状态（2026-07-11）

本文记录当前工作树中“说话人分析准确率提升方向”路线图的实现状态与验收证据。状态定义：

- **已达成**：代码、产品接入和对应验证均具备。
- **部分达成**：已有实现，但缺少端到端接入、真实评测或必要的闭环行为。
- **未达成**：路线图要求尚无对应实现或实现位置不匹配。

## 已验证基线

在当前工作树执行：

```powershell
uv run --frozen --extra dev pytest -q
$env:PYTHONUTF8='1'; python tools/verify_sample.py --all
python tools/run_regression.py score
python tools/build_regression_dashboard.py --output outputs/regression/acceptance-dashboard.json
node --check static/js/app.js
```

结果：pytest 20 项通过；样本校验通过（两份 bareflash 对照样本按设计给出豁免警告）；当前 11 个权威样本的整体准确率为 **92.86%**，简单场景 **95.0%**，密集场景 **87.0%**，`muli4_seg8` 为 **84.40%**。这些是既有快照的评分，不能证明新路由已带来提升。

## 路线图状态

| 条目 | 状态 | 当前证据与缺口 |
|---|---|---|
| 安全与工程基线 | 部分达成 | 输出目录 containment、上传限制、远程访问令牌、错误脱敏、锁文件和 CI 配置已加入；路径穿越和远程守卫已实测。BookVoiceParser 的 audit/SPC/BatchLLM 出站请求尚未统一接入 SSRF 校验。 |
| A1 密集场景路由到强模型 | 部分达成 | `/api/parse_v2` 支持可选 `dense_llm`，并有 `route_dense_to_llm()`；尚未接入解析 UI，也未用真实强模型重跑基准，未证明密集场景达到 90%+。 |
| A2 异构模型审计重问 | 部分达成 | 审计 API、前端异构配置和环境变量默认配置存在；缺少固定生产模型、成本策略及真实对照评测。 |
| C1 置信度重校准 | 部分达成 | 可生成单调分箱 artifact 并通过环境变量加载；默认关闭，没有提交正式 artifact、独立验证集或 ECE/Brier 指标。 |
| B1 生产级目标发现 | 部分达成 | 已实现多信号交集筛选；当前逻辑先排除“未知”说话人，导致 `unresolved` 信号无法覆盖这些段，需修正。 |
| B2 address_term 硬门控 | 部分达成 | 离线 `evaluate_evidence_review.py` 已增加门控；尚未统一覆盖全部在线复核路径。 |
| B4 链式传染检测 | 部分达成 | 改判后会返回前后邻段索引；尚未自动以修订后的上下文重审这些邻段。 |
| D1 跨类型基准 | 未达成 | 当前权威回归样本仍集中于《魔弹之王》系列，尚无现代都市、古风、玄幻等跨类型金标集。 |
| D2 回归仪表盘 | 已达成 | `tools/build_regression_dashboard.py` 可生成 JSON/HTML，汇总样本、密度、置信度桶和错误类型。 |
| D3 自动化测试 | 部分达成 | pytest 与 nightly 静态快照门禁已加入；尚未重跑外部模型的 5 样本夜间回归，且 Ruff 对相关文件仍有 lint 问题。 |
| E1 audit-safe 复核 UI | 未达成 | UI 展示 reask、flags、优先级与目标信号，但没有完整展示 `counter_evidence_type`、`baseline_evidence_valid`、`auto_apply_safe`、`reason`。 |
| E2 别名/身份预解析 | 部分达成 | 提供 `/api/character-aliases/preview`；尚未形成按书持久化、人工确认的身份注册表，也未接入 UI 工作流。 |
| E3 批量 404 重试 | 未达成 | BatchLLM HTTPX 分支增加了兼容路径尝试，但路线图指定的 `tools/evaluate_evidence_review.py` 仍未对 404 做指数退避重试。 |
| 主动学习闭环 | 部分达成 | 人工修订可落盘到本地 silver JSONL，支持去重、统计和导出；尚未有数据质检、周期训练、版本晋升和回灌评测。 |

## 当前提交范围

本次提交只包含可明确归属的安全、解析、评测、前端和文档改动。既有实验产物、临时脚本、样本快照与用户已有的场景摘要改动不混入提交。

## 后续优先级

1. 修正 B1 对未知段的筛选，并将 B2/B4 接入在线审计链路。
2. 为 A1/A2 选择受控的异构模型配置，重跑全量基准并记录成本、耗时、误伤率和准确率变化。
3. 建立 D1 跨类型金标集，并对 C1 做独立校准评测。
4. 完成 E1 结构化 audit-safe UI、E2 按书身份注册表、E3 指定工具的 404 指数退避。
5. 待 silver 样本积累后引入质量审核和版本化训练/回灌流程。
