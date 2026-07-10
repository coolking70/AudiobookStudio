# 主动学习数据闭环

人工修改说话人时，系统将修订记录追加到 `outputs/learning/silver.jsonl`。统计接口同时返回质量审计结果；缺失 ID、原文哈希、文本或说话人，以及未实际改变 speaker 的记录都会阻止版本晋升。

积累足够记录后执行：

```powershell
uv run --frozen python tools/export_silver_version.py --output outputs/learning/versions/silver-v1.json --min-records 20
```

导出包包含稳定 `dataset_id`、记录数、完整 SHA-256、质量报告和原始记录，可作为后续训练/回放评测的不可变输入。数据不足或质量门禁失败时命令直接失败，不产生可晋升版本。模型训练、候选版本评测与生产晋升仍需在有足够数据后由外部训练流水线执行。
