# IndexTTS2 生成效率实践结论

记录日期：2026-06-13

本文记录当前在本机环境中对 IndexTTS2 的参数、参考音频缓存和生成调度策略的实测结论。结论面向实践项目改造使用，生产接入前仍建议保留开关和回退路径。

## 推荐配置

当前综合听感和效率后，原稳定推荐参数如下：

```python
{
    "use_fp16": True,
    "use_cuda_kernel": False,
    "use_deepspeed": False,
    "use_torch_compile": True,
    "num_beams": 1,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 20,
    "max_text_tokens_per_segment": 120,
}
```

对应测试组为 `IndexTTS2-full-topk20`。完整 12 条日语参考音频生成中文台词的效率结果：

| 配置 | 成功数 | 总耗时 | 中位耗时 | 平均耗时 | 中位 RTF |
|---|---:|---:|---:|---:|---:|
| IndexTTS2 原始默认 | 12/12 | 未重跑 | 15.29s | 16.69s | 3.33 |
| `num_beams=1` | 12/12 | 93.24s | 7.43s | 7.77s | 1.58 |
| `torch_compile + num_beams=1` | 12/12 | 84.97s | 6.47s | 7.08s | 1.51 |
| `torch_compile + top_k=20` | 12/12 | 81.76s | 6.41s | 6.81s | 1.32 |

后续引擎层实验打开了 `do_sample`、`diffusion_steps`、`inference_cfg_rate` 的可调入口。完整 12 条候选结果如下：

| 配置 | 成功数 | 总耗时 | 中位耗时 | 平均耗时 | 中位 RTF |
|---|---:|---:|---:|---:|---:|
| `torch_compile + top_k=20` | 12/12 | 81.76s | 6.41s | 6.81s | 1.32 |
| `do_sample=False` | 12/12 | 63.57s | 4.87s | 5.30s | 1.22 |
| `do_sample=False + diffusion_steps=16` | 12/12 | 64.92s | 5.30s | 5.41s | 1.15 |
| `do_sample=False + diffusion_steps=16 + inference_cfg_rate=0.3` | 12/12 | 62.16s | 5.00s | 5.18s | 1.14 |

因此当前建议分两档：

- 稳定推荐：继续使用 `IndexTTS2-full-topk20`，即 `num_beams=1, temperature=0.8, top_p=0.8, top_k=20`。
- 实验提速候选：试听确认无退化后，可测试 `do_sample=False, diffusion_steps=16, inference_cfg_rate=0.3`。这组在当前样本中比稳定推荐总耗时约低 `24%`。

补充结论：

- `use_cuda_kernel=True` 在当前 Windows + CUDA 12.8 + PyTorch 2.8 环境中没有带来收益，测试中反而更慢，因此暂不建议默认启用。
- `torch_compile=True` 首次调用会有明显 warmup 开销，适合长会话、批量生成和常驻服务，不适合只生成一两句就退出的临时调用。
- `num_beams=1` 是目前最明显的提速项，用户听感上没有发现明显退化。
- `use_accel=True` 依赖 `flash_attn`。当前 Windows venv 中未安装，`I:\pip_cache` 中也没有可直接复用的 wheel，暂未验证。
- `do_sample=False` 会让 `temperature/top_p/top_k` 被 Transformers 生成逻辑忽略，属于确定性/贪心解码路径。它显著提速，但是否影响表现力需要人工听感确认。
- `diffusion_steps=16` 可降低 s2mel 计算量；与 `inference_cfg_rate=0.3` 组合目前最快，但音色贴合度需要听感复核后再作为生产默认。

## 最佳生成调度

当前最高效率实践不是严格按台词顺序逐句生成，而是：

1. 先保留原始 segment 顺序和 `segment_id`。
2. 按 `voice_id` 或参考音频路径把待生成台词分桶。
3. 对每个说话人连续生成全部片段。
4. 保存每段音频到带 `segment_id` 的临时文件。
5. 最后按原始 segment 顺序合并音频、生成 LRC 或时间轴。

原因是 IndexTTS2 源码中已有“最近一个参考音频”的单槽缓存。连续使用同一参考音频时，参考音频特征只需计算一次；如果按剧情顺序 A/B/A/B 交替生成，会频繁失效并重算参考音频特征。

推荐流水线：

```text
segments
  -> assign voice_id/ref_audio
  -> group by voice_id/ref_audio
  -> generate grouped audio clips
  -> collect {segment_id, clip_path, duration}
  -> sort by original segment order
  -> concatenate and write timeline
```

## 参考音频缓存结论

额外测试了实验性多参考缓存 adapter。固定随机种子后，单槽缓存和多参考缓存生成音频时长完全一致，结果如下：

| 场景 | 成功数 | 总耗时 | 中位耗时 | 平均耗时 | 中位 RTF |
|---|---:|---:|---:|---:|---:|
| 交替参考音频，原生单槽缓存 | 12/12 | 70.68s | 5.19s | 5.89s | 1.22 |
| 交替参考音频，多参考缓存 adapter | 12/12 | 65.23s | 5.02s | 5.44s | 1.13 |

参考音频纯预处理很轻，两个参考音频合计约 `0.33s`。因此：

- 离线整章生成：优先按说话人分组批量生成，通常比维护多参考缓存更简单、更高效。
- 实时预览、局部重生成、边读边生成：可以考虑多参考缓存，减少角色切换时的重复特征提取。
- 生产接入建议使用环境变量开关，例如 `INDEX_TTS_ENABLE_REF_CACHE=1`，默认关闭，方便回退。

## 实践项目改造建议

建议把生成任务从“逐段顺序调用 TTS”改为“生成计划 + 分组执行 + 顺序整合”：

```python
jobs = [
    {
        "segment_id": segment.id,
        "order": index,
        "speaker": segment.speaker,
        "voice_id": assigned_voice.id,
        "ref_audio": assigned_voice.ref_audio,
        "text": segment.text,
        "output_path": clip_path,
    }
    for index, segment in enumerate(segments)
]

for ref_audio, group in groupby_ref_audio(jobs):
    for job in group:
        generate_one(job)

clips = sorted(jobs, key=lambda job: job["order"])
merge_audio([job["output_path"] for job in clips])
```

注意事项：

- 合并阶段必须依赖原始 `order`，不要依赖文件名排序。
- 旁白、未知说话人和未分配音色应统一落到稳定的默认 voice_id，避免频繁产生新桶。
- 如果用户在 UI 中请求“从当前位置开始试听”，实时路径可以继续按顺序生成，但整章导出路径建议使用分组生成。
- 如果后续接入多进程或多 GPU，可以在 `voice_id` 分桶基础上进一步调度并行，但单 GPU 下不要同时跑多个 IndexTTS2 实例，容易引入显存压力和 compile/cache 抖动。

## 当前实验文件

相关实验脚本与输出：

- `bench_scripts/run_indextts2_selected_params_full_japanese_sample_lines.py`
- `bench_scripts/run_indextts2_engine_knob_matrix.py`
- `bench_scripts/run_indextts2_greedy_combo_matrix.py`
- `bench_scripts/run_indextts2_engine_candidates_full.py`
- `bench_scripts/run_indextts2_prepare_ref_cache_probe.py`
- `bench_scripts/indextts2_cached_adapter.py`
- `bench_outputs/japanese_ref_to_chinese_sample_lines_20260612/listen.html`
