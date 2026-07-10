# 密集多人场景异构路由

`POST /api/parse_v2` 支持可选的 `dense_llm` 配置。启用后，系统先按现有主流水线完成解析，再检测局部窗口内至少出现 3 名具名说话人的片段，仅将这些片段交给 `dense_llm` 指定模型复核。

```json
{
  "text": "…",
  "role_hints": {"角色甲": [], "角色乙": [], "角色丙": []},
  "use_batch_llm": true,
  "llm": {"base_url": "http://127.0.0.1:1234/v1", "api_key": "local", "model": "baseline"},
  "dense_llm": {"base_url": "https://provider.example/v1", "api_key": "…", "model": "strong-review-model"}
}
```

该路由默认关闭，不会改变旧请求行为。结果中的 `stats.dense_model_route` 包含目标数量、目标索引和证据门控统计；模型建议未通过既有安全门控时只进入人工复核证据，不会强行覆盖基线。

审计接口 `/api/audit_segments/stream` 默认使用 `target_mode=production`：只有低置信度、密集场景、候选冲突、已有复核证据、未解析标签等信号至少两项同时出现时才进入重问队列。需要完整扫描时可显式传 `target_mode=all`。

如需为审计配置默认异构模型，可设置 `AUDIOBOOKSTUDIO_HETERO_AUDIT_BASE_URL`、`AUDIOBOOKSTUDIO_HETERO_AUDIT_MODEL` 和可选的 `AUDIOBOOKSTUDIO_HETERO_AUDIT_API_KEY`；请求体中的 `hetero_llm` 优先级更高。
