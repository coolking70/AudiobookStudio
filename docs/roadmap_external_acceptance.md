# 路线图外部验收清单

以下目标不能由仓库内代码或既有单一作品快照可靠证明，必须使用真实模型服务或新增人工金标。不要用模拟结果将其标记为达成。

## A1 / A2 模型效果

1. 固定主管线、密集强模型与异构审计模型的名称、版本、参数和日期。
2. 对同一批权威样本分别运行基线、`dense_llm`、`dense_llm + hetero_llm`，保存原始输出、调用次数、token/费用和耗时。
3. 执行 `uv run --frozen python tools/run_regression.py score`，并生成回归仪表盘。
4. A1 的晋升门槛：密集场景准确率至少 90%，整体准确率不下降，简单场景误伤率不增加；A2 还需记录异构扫描的净修正、误伤与单样本成本。

## D1 跨类型金标

至少新增现代都市、古风、玄幻三类作品，每类包含简单与三人以上密集对话；人工双人复核 speaker/acceptable 后再提交 groundtruth。所有新样本必须通过：

```powershell
$env:PYTHONUTF8='1'
uv run --frozen python tools/verify_sample.py --all
uv run --frozen python tools/run_regression.py score
```

## 主动学习训练晋升

当有效人工修订达到约定规模后，先导出不可变数据版本：

```powershell
uv run --frozen python tools/export_silver_version.py --output outputs/learning/versions/silver-v1.json --min-records 20
```

训练候选模型必须记录输入 dataset ID/SHA-256，并在冻结回归集上优于当前生产模型且无关键场景回退，才可晋升；失败版本保留报告但不得覆盖生产配置。
