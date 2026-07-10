# 路线图实施与验收状态（2026-07-11）

本文记录当前工作树中“说话人分析准确率提升方向”路线图的实现状态与验收证据。状态定义：

- **已达成**：代码、产品接入和对应验证均具备。
- **部分达成**：已有实现，但缺少端到端接入、真实评测或必要的闭环行为。
- **未达成**：路线图要求尚无对应实现或实现位置不匹配。

## 已验证基线

在当前工作树执行：

```powershell
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check security.py schemas.py llm_client.py project_store.py confidence_calibration.py learning_store.py tests BookVoiceParser/book_voice_parser/security_bridge.py BookVoiceParser/book_voice_parser/audit.py BookVoiceParser/book_voice_parser/review_router.py BookVoiceParser/book_voice_parser/spc_ranker.py
$env:PYTHONUTF8='1'; python tools/verify_sample.py --all
python tools/run_regression.py score
python tools/build_regression_dashboard.py --output outputs/regression/acceptance-dashboard.json
node --check static/js/app.js
```

结果：pytest **36 项通过**，安全与维护核心的 Ruff 检查通过，Python 编译与两份前端脚本语法检查通过；浏览器冒烟验证确认身份预解析入口、密集强模型面板和页面运行日志正常。样本校验通过（两份 bareflash 对照样本按设计给出豁免警告）；当前 11 个权威样本的整体准确率为 **92.86%**，简单场景 **95.0%**，密集场景 **87.0%**，`muli4_seg8` 为 **84.40%**。这些是既有快照的评分，不能证明新路由已带来提升。

## 路线图状态

| 条目 | 状态 | 当前证据与缺口 |
|---|---|---|
| 安全与工程基线 | 已达成 | 输出目录 containment、上传与全局请求体限制（含 chunked）、远程访问令牌、错误脱敏、锁文件和 CI 门禁已加入；所有已知 OpenAI 兼容出站路径（主客户端、BatchLLM、SPC、audit）统一执行 SSRF/DNS fail-closed 校验并禁止重定向。API 级回归覆盖路径穿越、内网/元数据地址、DNS 失败、超限请求体和 schema 数量/长度上限。 |
| A1 密集场景路由到强模型 | 部分达成 | `/api/parse_v2` 与 SSE BatchLLM 主管线均支持 `dense_llm`，解析 UI 已提供独立强模型开关、配置与本地持久化；尚未用真实强模型重跑基准，未证明密集场景达到 90%+。 |
| A2 异构模型审计重问 | 部分达成 | 审计 API、前端异构配置和环境变量默认配置存在；缺少固定生产模型、成本策略及真实对照评测。 |
| C1 置信度重校准 | 已达成 | 已提交晋升 artifact `calibration/confidence_calibrator_v1.json`。用 2,214 条记录拟合，在按样本隔离的 501 条验证集上，Brier 由 0.12969062 降至 0.10149758，ECE 由 0.13668663 降至 0.02322862；拟合工具只有两项均不劣化才标记 `promoted=true`，生产仍显式启用以便灰度回滚。 |
| B1 生产级目标发现 | 已达成 | 生产审计执行多信号交集筛选；未知/临时人物不再被提前过滤，可由 `low_confidence + unresolved` 等交集进入目标队列，并有回归测试。 |
| B2 address_term 硬门控 | 已达成 | 离线 evidence apply 与在线 SPC/Batch 复核统一禁止仅凭称呼关系自动改写；只有动作主语、第一人称或前后文明示说话人时才允许继续通过其他安全门控。 |
| B4 链式传染检测 | 已达成 | Batch 复核发生自动改判后会强制以更新后的段列表重审 ±1 邻段；传播限制为一层并返回 `chain_review_indices/chain_review` 统计，避免调用递归放大。 |
| D1 跨类型基准 | 未达成 | 当前权威回归样本仍集中于《魔弹之王》系列，尚无现代都市、古风、玄幻等跨类型金标集。 |
| D2 回归仪表盘 | 已达成 | `tools/build_regression_dashboard.py` 可生成 JSON/HTML，汇总样本、密度、置信度桶和错误类型。 |
| D3 自动化测试 | 部分达成 | pytest、维护核心 Ruff 门禁与 nightly 静态快照门禁已加入；尚未重跑外部模型的 5 样本夜间回归，仓库中历史实验脚本的全量 lint 债务也尚未清理。 |
| E1 audit-safe 复核 UI | 已达成 | 审计提示词与工作台完整携带并展示 `counter_evidence_type`、`baseline_evidence_valid`、`auto_apply_safe`、`reason`，同时保留 reask、flags、优先级与目标信号。 |
| E2 别名/身份预解析 | 已达成 | UI 可在主管线前调用 `/api/character-aliases/preview`，将结果转成可人工编辑确认的角色 JSON；确认映射会随统一项目角色表持久化，项目页支持继续编辑别名并用于 canonical speaker 解析。 |
| E3 批量 404 重试 | 已达成 | `tools/evaluate_evidence_review.py` 的在线调用已接入有界 404 指数退避（0.5s/1s/2s，非 404 不重试），并有无真实等待的单元测试。 |
| 主动学习闭环 | 部分达成 | 人工修订可去重落盘到 silver JSONL，统计接口包含质量审计；版本导出执行必填字段/重复/未改判门禁并生成稳定 dataset ID 与 SHA-256。尚需在真实 silver 数据达到规模后运行外部训练、候选模型回放评测和生产晋升。 |

## 当前提交范围

本次提交只包含可明确归属的安全、解析、评测、前端和文档改动。既有实验产物、临时脚本、样本快照与用户已有的场景摘要改动不混入提交。

## 后续优先级

1. 安全基线、B1/B2/B4 与 E1/E2/E3 已完成；下一步为 A1/A2 选择受控的异构模型配置，重跑全量基准并记录成本、耗时、误伤率和准确率变化。
2. 建立 D1 跨类型金标集；C1 独立校准评测与正式 artifact 晋升已完成。
3. 重跑外部模型的 5 样本夜间回归，清理仓库历史实验脚本的全量 lint 债务。
4. 待 silver 样本积累后引入质量审核和版本化训练/回灌流程。
