"""A/B benchmark: IndexTTS2 use_accel (flash_attn accel engine) vs baseline.

Mirrors run_indextts2_engine_candidates_full.py exactly (same refs, same target
texts, same greedy variants, use_torch_compile=True), but runs the 3 greedy
variants twice in one session:
  - base  : use_accel=False  (matches the documented IndexTTS2-full-greedy* runs)
  - accel : use_accel=True   (GPT stage driven by the self-built flash_attn engine)

The accel engine only honours `temperature` for the GPT stage (top_p/top_k/
do_sample/repetition_penalty/num_beams are ignored by accel_engine.generate), so
the greedy family is the fair comparison. diffusion_steps / inference_cfg_rate are
s2mel knobs and still apply in both modes.

Running both modes in the same session controls for machine state / run-to-run
variance (the docs note ~±3-9% parse-time variance; keep that in mind).
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

BASE = {"num_beams": 1, "do_sample": False, "temperature": 0.8, "top_p": 0.8, "top_k": 20}
VARIANT_SUFFIXES = [
    ("greedy", {**BASE}),
    ("greedy-diff16", {**BASE, "diffusion_steps": 16}),
    ("greedy-diff16-cfg03", {**BASE, "diffusion_steps": 16, "inference_cfg_rate": 0.3}),
]

# (engine_prefix, use_accel)
MODES = [
    ("IndexTTS2-base", False),
    ("IndexTTS2-accel", True),
]


def run_mode(IndexTTS2, refs, targets, engine_prefix: str, use_accel: bool) -> None:
    print(f"\n==== build model: {engine_prefix} (use_accel={use_accel}) ====", flush=True)
    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=use_accel,
        use_torch_compile=True,
    )

    warmup_path = OUTPUT_ROOT / f"_indextts2_accel_ab_warmup_{engine_prefix}.wav"
    print(f"{engine_prefix} warmup start", flush=True)
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        num_beams=1,
    )
    print(f"{engine_prefix} warmup done", flush=True)

    for suffix, infer_kwargs in VARIANT_SUFFIXES:
        engine = f"{engine_prefix}-{suffix}"
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
                        **infer_kwargs,
                    )

                ok, seconds, duration, rtf, error = timed_generate(call, output_path)
                rows.append(
                    ResultRow(
                        engine, ref["voice_id"], ref["speaker"], phrase_id, text, 1,
                        ok, str(output_path) if ok else "", seconds, duration, rtf, error,
                    )
                )
                write_manifest(rows, out_dir)
                print(f"{engine} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)

    # free GPU memory before building the next model
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    # match the cache paths used by the prior documented runs
    os.environ["HF_HOME"] = r"I:\hf_cache"
    os.environ["HF_HUB_CACHE"] = r"I:\hf_cache"
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    refs = prepare_ascii_refs(OUTPUT_ROOT / "_indextts2_accel_ab_refs")
    targets = extract_target_texts()
    print(f"refs={len(refs)} targets={len(targets)} -> {len(refs)*len(targets)} gens per variant", flush=True)

    for engine_prefix, use_accel in MODES:
        run_mode(IndexTTS2, refs, targets, engine_prefix, use_accel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
