# 置信度校准（C1）

解析器原始 `confidence` 现在支持可选的、可回滚的分箱校准。校准默认关闭；设置 `AUDIOBOOKSTUDIO_CONFIDENCE_CALIBRATOR` 指向 JSON artifact 后才会启用。

从已提交的 parse/groundtruth 样本拟合：

```powershell
python tools/fit_confidence_calibrator.py --output outputs/regression/confidence_calibrator.json
$env:AUDIOBOOKSTUDIO_CONFIDENCE_CALIBRATOR = "outputs/regression/confidence_calibrator.json"
```

artifact 使用单调 isotonic 分箱表，记录版本、样本量、每箱样本数和校准值。加载失败时解析器安全降级为原始 confidence；成功应用后会在 `stats.confidence_calibration` 记录 artifact 路径和变更数量，并在片段证据中追加 `confidence_calibrated` 标记。

仓库已晋升 `calibration/confidence_calibrator_v1.json`：用 2,214 条训练记录拟合，在按样本隔离的 501 条验证集上，Brier 从 0.12969062 降至 0.10149758，ECE 从 0.13668663 降至 0.02322862。只有两项指标均不劣于原始置信度时，拟合工具才会写入 `promoted: true`；生产启用仍需显式设置环境变量，便于灰度与回滚。
