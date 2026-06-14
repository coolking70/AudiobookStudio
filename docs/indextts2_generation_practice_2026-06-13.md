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
- `use_accel=True` 依赖 `flash_attn`。已在 Windows venv 中自行编译安装成功，`import flash_attn` 与 `indextts.accel` 导入链均通过，GPU 实测前向计算正常（详见下文「flash_attn 自编译记录」）。提速收益仍待用 IndexTTS2 实际生成基准测试。
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

## flash_attn 自编译记录（2026-06-13 解决）

`use_accel=True` 需要 `flash_attn`，但官方只发 Linux 包，Windows 无官方预编译。结论：**社区预编译 wheel 不可用，自行编译成功并已可用。**

### 为什么社区 wheel 不行

`lldacing/flash-attention-windows-wheel`（hf-mirror）里唯一匹配的
`flash_attn-2.7.4.post1+cu128torch2.8.0cxx11abiTRUE-cp310-cp310-win_amd64.whl`
能 `pip install`，但 `import` 时报
`DLL load failed while importing flash_attn_2_cuda: 找不到指定的程序`（WinError 127，符号缺失）。
根因是该 wheel 链接的 torch ABI 与本机 `torch 2.8.0+cu128` 不一致。该 wheel 已改名
`*.BROKEN-abi-mismatch` 标记不可用。

### 成功的自编译方法（针对本机 .venv 的 torch 直接编译，ABI 必然匹配）

环境：Win11 / Python 3.10 / torch 2.8.0+cu128 / RTX 4070 Ti SUPER (compute 8.9) / 64GB RAM / 16 核。

1. **CUDA 12.8 toolkit**：用 conda 装进 I 盘，不碰 C 盘、走清华镜像，避免大安装包：
   `conda create -n fa-build -c <tuna conda-forge> --override-channels -y "cuda-toolkit=12.8.*"`
   （nvcc 在 `I:\conda_envs\fa-build\Library\bin`）
2. **.venv 编译依赖**：`uv pip install ninja packaging psutil setuptools wheel`（清华 PyPI）
3. **源码**：`git clone --depth 1 --branch v2.7.4.post1 --recurse-submodules --shallow-submodules`，
   github 经 `ghfast.top` CDN 加速（用 `git config --global url.<cdn>.insteadOf` 临时改写，用完还原）。
4. **编译关键点**（脚本 `I:\fa-build-src\build_flash_attn.bat`，先 `call vcvars64.bat`）：
   - 直接用 index-tts 的 `.venv\Scripts\python.exe` 跑 `setup.py bdist_wheel` → 产出 wheel 与目标 torch ABI 一致
   - `CUDA_HOME=CUDA_PATH=I:\conda_envs\fa-build\Library`
   - `FLASH_ATTN_CUDA_ARCHS=80`：flash-attn 2.7.4 的 gencode 只支持 80/90/100/120，**没有 89**；
     Ada(8.9) 与 Ampere(8.0) 同属 8.x，**sm_80 cubin 可直接在 8.9 上运行**（已 GPU 实测）。只编 sm_80 最快。
   - `set INCLUDE=...\Library\include\targets\x64;%INCLUDE%`：conda-forge 把 CCCL 头（`nv/target`、`cub`、`cuda/std`）放这，否则 `cuda_fp16.h` 报 `C1083: nv/target`。
   - `set LIB=...\Library\lib;%LIB%`：conda-forge 的 `cudart.lib` 在 `Library\lib`（torch 默认找 `lib\x64`），否则 `LNK1181: cudart.lib`。
   - `set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`：本机 MSVC 14.50(VS18) 比 CUDA 12.8 支持上限(VS2022)新，需绕过 `host_config.h` 版本检查（实测 14.50 能正常编译）。
   - `MAX_JOBS=4`：bwd hdim128/192 反向核编译极吃内存，12 路并行会 `catastrophic error: out of memory`；4 路稳定（64GB 下）。
5. 产物 `flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl`（88.5MB），已备份到
   `I:\pip_cache\manual_wheels\flash_attn-2.7.4.post1-cp310-cp310-win_amd64-SELFBUILT-torch2.8.0-cu128-sm80.whl`，
   日后同环境可直接复用，无需重编。

### 验证

- `import flash_attn` / `flash_attn_varlen_func` / `flash_attn_with_kvcache`：OK
- `from indextts.accel import GPT2AccelModel, AccelInferenceEngine`：OK
- GPU 实跑 `flash_attn_func`（fp16, causal）：输出正常，sm_80 在 Ada 8.9 上无 kernel 错误。

> 注意：若日后 `.venv` 升级 torch（ABI 变化），需用相同方法重编。`fa-build` conda 环境与
> `I:\fa-build-src` 源码可保留以便快速重编。

## use_accel 实测基准（flash_attn 加速引擎，2026-06-13）

flash_attn 自编译可用后，跑了一次同会话 A/B（控制机器状态/run-to-run 方差），脚本
`bench_scripts/run_indextts2_accel_japanese_sample_lines.py`。同一组日语参考音频 ×
中文台词（2 refs × 6 句 = 12 条/变体），`use_fp16=True, use_torch_compile=True`，
仅翻转 `use_accel`，对 3 个 greedy 变体各跑一遍。

> 为什么只对 greedy 系列：`accel_engine.generate` 只接受 `temperature`，会忽略
> `top_p/top_k/do_sample/repetition_penalty/num_beams`（见 `model_v2.py` inference_speech），
> 且仅在 `num_return_sequences==1` 时启用（`autoregressive_batch_size` 硬编码为 1，满足）。
> `diffusion_steps/inference_cfg_rate` 是 s2mel 阶段参数，两种模式都生效。
> `use_torch_compile` 只编译 s2mel，不碰 GPT，与 accel 正交、可共存。

| 配置 | 成功数 | 总耗时 | 中位耗时 | 平均耗时 | 中位 RTF |
|---|---:|---:|---:|---:|---:|
| base greedy（accel=off） | 12/12 | 71.12s | 5.73s | 5.93s | 1.318 |
| base greedy+diff16 | 12/12 | 63.52s | 4.85s | 5.29s | 1.169 |
| base greedy+diff16+cfg03 | 12/12 | 65.63s | 5.04s | 5.47s | 1.223 |
| **accel greedy（accel=on）** | 12/12 | **21.60s** | **1.65s** | 1.80s | **0.352** |
| **accel greedy+diff16** | 12/12 | **19.18s** | **1.37s** | 1.60s | **0.315** |
| **accel greedy+diff16+cfg03** | 12/12 | **20.09s** | **1.53s** | 1.67s | **0.305** |

交叉校验：本次 base 三档（4.85–5.73s）与 2026-06-12 文档旧 run 的 `IndexTTS2-full-greedy*`
（4.87–5.30s）一致，落在文档所述 ±方差内，说明口径与机器状态可比。

**结论：`use_accel=True` 在本机（RTX 4070 Ti SUPER / torch 2.8 / 自编译 flash_attn sm_80）
带来约 3.3× 端到端提速**（中位 5.0s → 1.5s，中位 RTF 约 1.2 → 0.3），12/12 全部成功、无失败。
与 `diffusion_steps=16 + inference_cfg_rate=0.3` 叠加后最快（中位 1.53s）。

注意事项与后续：
- accel 走贪心/温度采样路径，表现力是否与采样路径（top_k=20 等）有差异需人工听感复核；
  本次只验证了效率与稳定性，未做听感对比。试听页：上述 OUTPUT_ROOT 下 `listen.html`
  （`IndexTTS2-accel-*` 与 `IndexTTS2-base-*` 可直接 A/B 切换）。
- accel 引擎构建有额外初始化开销（CUDA graph + 自有 KV cache），适合常驻服务/批量；
  一次性短调用收益会被初始化摊薄。
- 生产接入建议保留 `use_accel` 开关与非 accel 回退路径。

## use_cuda_kernel + use_accel 组合复测（2026-06-13）

复核"`use_cuda_kernel` 无收益"的旧结论是否在叠加 accel 后改变。脚本
`bench_scripts/run_indextts2_accel_cudakernel_ab.py`，固定 `use_accel=True`，仅翻转
`use_cuda_kernel`，跑推荐最快配置（greedy + diffusion_steps=16 + inference_cfg_rate=0.3），
同会话各 12 条。

### 先踩到的坑：cuda_kernel 会"静默回退"

`use_cuda_kernel=True` 时 BigVGAN 会用 `torch.utils.cpp_extension.load` 加载融合核。
在 Windows 上，**即便预编译 .pyd 已存在，load 也要求运行时 PATH 上有 `ninja` 和
`cl.exe`（MSVC）**，否则抛异常并被 `infer_v2.py` 静默回退到 torch
（`model.use_cuda_kernel` 被悄悄改成 False）。前两次直接用 `.venv\python.exe` 跑都因此回退
（先缺 ninja，后缺 cl.exe），看似"开了"其实没开。必须在 `vcvars64 + ninja + nvcc` 完整
环境里跑，才真正加载（`>> Preload custom CUDA kernel for BigVGAN ...pyd`，
`use_cuda_kernel after init = True`）。运行脚本见 `I:\fa-build-src\run_accel_cudakernel_ab.bat`。

### 真正加载后的实测

| 配置（均 use_accel=True） | 成功 | 总耗时 | 中位耗时 | 平均耗时 | 中位 RTF | bigvgan 中位 |
|---|---:|---:|---:|---:|---:|---:|
| kernel-off（torch 声码器） | 12/12 | 17.52s | 1.40s | 1.46s | 0.288 | 0.130s |
| kernel-on（CUDA 融合核） | 12/12 | 17.65s | 1.32s | 1.47s | 0.282 | 0.060s |

**结论：CUDA 融合核确实把 BigVGAN 声码阶段提速约 2×（0.13s→0.06s），但端到端无意义。**
原因是 BigVGAN 只占整条流水线很小一块（accel 把 GPT 压到 ~0.03s 后，单句总时 ~1.4s 中
bigvgan 仅 ~0.13s，约 9%）。省下的 ~0.07s 落在 run-to-run 方差内——两者平均总耗时几乎相同
（1.46 vs 1.47s）。

修正旧结论：旧表"`use_cuda_kernel=True` 更慢"的判断，很可能也受"静默回退/预编译核 arch 非最优"
干扰。**干净复测表明：核本身并不更慢，真正加载后阶段级是正向的；但端到端收益被 BigVGAN 的
小占比稀释到噪声内。**

### 是否值得用

不建议在生产开启。理由：
- 端到端无可测收益（在噪声内）。
- **运维脆弱**：必须在 `vcvars64 + ninja + CUDA toolkit` 齐备的环境启动才会真正加载，否则静默
  回退——普通 `.venv` 启动（无 vcvars）必然回退，还会误以为已开启。
- 收益上限就是 bigvgan 那 ~0.07s，不值得这套运行时编译依赖。

真正的提速来源仍是 `use_accel`（GPT 阶段，~3.3×）与 `num_beams=1 / do_sample=False /
diffusion_steps=16 / inference_cfg_rate=0.3`。

## use_cuda_kernel 复核补遗：脆弱性可解 + 是否值得（2026-06-14 更新）

针对"数千条对话的批量场景，即便微小提升也值得"的诉求，进一步确认了两点：脆弱性能否解决、有无其它收益。

### 脆弱性已彻底解决（patch load.py）

根因：`bigvgan/.../cuda/load.py` 无脑走 `torch.utils.cpp_extension.load()`，而它在 Windows 上
**即便加载已编译好的 .pyd 也要求运行时 PATH 有 ninja + cl.exe(MSVC)**，否则抛异常→静默回退。

实测：预编译的 `anti_alias_activation_cuda.pyd` 其实能在**裸 .venv（无 vcvars/ninja/cl.exe）下直接
`import` 成功**（只依赖 torch DLL，和 flash_attn 一样）。据此给 `load.py` 加了 `_try_load_prebuilt()`
快路径：**优先直接 import 预编译产物，失败才回退到原 JIT 编译逻辑**。

验证：打 patch 后用裸 .venv 重跑，kernel-on 模式
`>> Preload custom CUDA kernel for BigVGAN ...pyd`、`use_cuda_kernel after init = True`，
24/24 成功，不再回退。**生产部署只需带上该 .pyd，零构建工具依赖。**
（注意：若日后升级 torch 导致 ABI 变化，直接 import 会失败并回退/需删 build 目录重编。）

### 是否有其它收益：显存——无

本次测量峰值显存（warmup 后 reset）：kernel-off 8506.5 MB vs kernel-on 8502.8 MB，**基本相同**。
BigVGAN 中间张量相对 GPT/s2mel 权重 + accel KV cache 太小，融合核不改变峰值显存，
**不能换取更大批量的并行余量**。所以唯一的实际效果就是 bigvgan 阶段本身提速。

### 速度：阶段级可靠 2×，端到端小幅正向（不再更慢）

两次独立运行一致：

| | bigvgan 中位 | 端到端中位 | 端到端均值 | 峰值显存 |
|---|---:|---:|---:|---:|
| kernel-off | 0.12–0.13s | 1.40s | 1.46–1.56s | ~8506 MB |
| kernel-on | 0.06s | 1.32–1.34s | 1.47–1.54s | ~8503 MB |

bigvgan 阶段稳定省 ~0.06s/条（可靠 2×）。端到端单条在方差内但偏正向，**不再像旧结论那样"更慢"**
（旧"更慢"应是静默回退/预编译核 arch 非最优的假象）。

### 对批量场景的结论（修正旧建议）

- **数千条批量：建议开启。** patch 后加载可靠、显存无副作用、端到端不再更慢，bigvgan 的
  ~0.06s/条是真实可靠节省，3000 条 ≈ 省 ~3 分钟，属"免费"小收益。
- 仍无显存/批量并行收益，主提速来源依旧是 `use_accel`（GPT，~3.3×）。
- 部署要点：① 用 patch 后的 `load.py`；② 随包携带 `.../cuda/build/anti_alias_activation_cuda.pyd`；
  ③ 升级 torch 后需重编该 .pyd。
- 建议对 kernel-on 输出做一次听感抽查（融合核与 torch 路径数值略有差异，虽是同一抗混叠激活数学）。

## 流水线接入：accel 配置 + 常驻服务（2026-06-14）

把前述优化结论落地到 omnivoice-reader 实际生成流水线，全部带开关与回退。

### 1. 加速配置接入（index_tts_bridge.py + app.py）

- `use_accel` 进 load options，默认开（env `INDEX_TTS_USE_ACCEL`）；flash_attn 不可用时
  `_load_model` 自动回退 `use_accel=False` 并告警，返回 `use_accel_effective` 供上层查看。
- 推荐提速旋钮进 generation options：`do_sample=False / diffusion_steps=16 /
  inference_cfg_rate=0.3`（均 env 可覆盖）。
- `app.py` 的 `INDEX_TTS_RECOMMENDED_*` 常量与 effective getter 同步更新，并在
  `build_index_tts_bridge_env` 把 effective 选项注入 bridge 子进程 → app 层为单一配置源。

### 2. 常驻服务（最关键的批量优化）

**问题**：原 `run_index_tts_bridge` 每段 `subprocess.run` 新起进程，每段都重载 ~6GB 模型 +
重跑 torch.compile/accel warmup。实测**冷启动单段 tts = 62.5s**，而进程内热生成仅 ~1.5s。
即模型内单槽参考缓存随进程销毁、分组逻辑零收益。3000 段 ≈ 52 小时（几乎全耗在重载）。

**方案**：给 bridge 加 `serve` 常驻模式（加载一次，按行收发 JSON）；app.py 增加常驻进程管理
（`run_index_tts_bridge_serve` + 统一入口 `run_index_tts_bridge_tts`，持久化优先、失败自动
回退单次调用）。narration 批量与单次预览都走统一入口（避免双份显存）。env `INDEX_TTS_PERSISTENT`
默认开。stderr 重定向到日志文件防止管道死锁；后台 reader 线程 + 队列收响应；atexit 清理。

**实测（serve 模式连发 3 段）**：

| 请求 | 耗时 | 说明 |
|---|---:|---|
| req#1 | 67.5s | 含模型加载 + 编译 warmup |
| req#2 | 1.2s | 模型已驻留 |
| req#3 | 1.6s | 相同参考音频 |

首段之后 ~1.5s/段，约 **40× 提速**。3000 段：~52 小时 → **~76 分钟**。配合已有的
`synthesize_index_tts_narration` 按 ref_audio 分组（之前因进程隔离无效，现在常驻后单槽缓存
跨段存活，分组真正生效）。

### 开关一览（env，全部可回退）

| 变量 | 默认 | 作用 |
|---|---|---|
| `INDEX_TTS_USE_ACCEL` | 1 | flash_attn 加速引擎 |
| `INDEX_TTS_PERSISTENT` | 1 | 常驻服务（批量关键） |
| `INDEX_TTS_USE_CUDA_KERNEL` | 0 | BigVGAN 融合核（批量可选） |
| `INDEX_TTS_DO_SAMPLE` | 0 | 贪心解码 |
| `INDEX_TTS_DIFFUSION_STEPS` | 16 | s2mel 步数 |
| `INDEX_TTS_INFERENCE_CFG_RATE` | 0.3 | s2mel cfg |

验证：两文件语法 OK；app.py 在 conda omnivoice 环境导入 OK；serve 模式三段实测模型仅加载一次、
accel 全程生效；flash_attn 回退逻辑单测通过。
