"""Focused A/B: with use_accel=True fixed, toggle use_cuda_kernel.

Question: the BigVGAN fused CUDA kernel was judged "no benefit / slower" WITHOUT
accel. Does combining it WITH accel change the verdict?

accel speeds up the GPT stage; use_cuda_kernel only affects the BigVGAN vocoder
stage. They are orthogonal pipeline stages, so combining just adds their
individual effects. This runs the recommended fastest config
(greedy + diffusion_steps=16 + inference_cfg_rate=0.3) twice in one session,
toggling only use_cuda_kernel, to measure empirically (controls machine state).
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

from common_japanese_sample_lines import (
    OUTPUT_ROOT,
    ResultRow,
    extract_target_texts,
    prepare_ascii_refs,
    timed_generate,
    write_manifest,
)


INDEX_ROOT = Path(r"I:\code\aitts\index-tts")
MODEL_DIR = INDEX_ROOT / "checkpoints"

INFER = {"num_beams": 1, "do_sample": False, "temperature": 0.8, "top_p": 0.8,
         "top_k": 20, "diffusion_steps": 16, "inference_cfg_rate": 0.3}

# (engine, use_cuda_kernel) — use_accel=True fixed for both
MODES = [
    ("IndexTTS2-accel-cfg03-kernel-off", False),
    ("IndexTTS2-accel-cfg03-kernel-on", True),
]


def run_mode(IndexTTS2, refs, targets, engine: str, use_cuda_kernel: bool) -> None:
    print(f"\n==== build model: {engine} (use_accel=True, use_cuda_kernel={use_cuda_kernel}) ====", flush=True)
    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=False,
        use_accel=True,
        use_torch_compile=True,
    )
    print(f"{engine}: model.use_cuda_kernel after init = {model.use_cuda_kernel}", flush=True)

    warmup_path = OUTPUT_ROOT / f"_accel_cudakernel_warmup_{engine}.wav"
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        num_beams=1,
    )
    print(f"{engine} warmup done", flush=True)

    # peak VRAM during the measured run (reset after warmup so warmup compile isn't counted)
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    out_dir = OUTPUT_ROOT / engine
    rows: list[ResultRow] = []
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                model.infer(
                    spk_audio_prompt=ref["ref_audio_ascii"],
                    text=text,
                    output_path=str(output_path),
                    verbose=False,
                    max_text_tokens_per_segment=120,
                    **INFER,
                )

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(
                ResultRow(engine, ref["voice_id"], ref["speaker"], phrase_id, text, 1,
                          ok, str(output_path) if ok else "", seconds, duration, rtf, error)
            )
            write_manifest(rows, out_dir)
            print(f"{engine} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)

    try:
        import torch
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{engine}: peak_vram_mb={peak:.1f}", flush=True)
    except Exception:
        pass

    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    os.environ["HF_HOME"] = r"I:\hf_cache"
    os.environ["HF_HUB_CACHE"] = r"I:\hf_cache"
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    refs = prepare_ascii_refs(OUTPUT_ROOT / "_accel_cudakernel_refs")
    targets = extract_target_texts()
    print(f"refs={len(refs)} targets={len(targets)} -> {len(refs)*len(targets)} gens per mode", flush=True)

    for engine, use_cuda_kernel in MODES:
        run_mode(IndexTTS2, refs, targets, engine, use_cuda_kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
