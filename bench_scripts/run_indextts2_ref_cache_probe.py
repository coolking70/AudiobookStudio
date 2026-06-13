from __future__ import annotations

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
TARGET_IDS = {"sample01", "sample04"}
INFER_KWARGS = {
    "num_beams": 1,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 20,
}


def _capture_reference_cache(model, ref_audio_prompt: str) -> dict:
    return {
        "cache_spk_cond": model.cache_spk_cond,
        "cache_s2mel_style": model.cache_s2mel_style,
        "cache_s2mel_prompt": model.cache_s2mel_prompt,
        "cache_mel": model.cache_mel,
        "cache_spk_audio_prompt": ref_audio_prompt,
        "cache_emo_cond": model.cache_emo_cond,
        "cache_emo_audio_prompt": ref_audio_prompt,
    }


def _restore_reference_cache(model, cache: dict) -> None:
    for key, value in cache.items():
        setattr(model, key, value)


def _clear_reference_cache(model) -> None:
    model.cache_spk_cond = None
    model.cache_s2mel_style = None
    model.cache_s2mel_prompt = None
    model.cache_mel = None
    model.cache_spk_audio_prompt = None
    model.cache_emo_cond = None
    model.cache_emo_audio_prompt = None


def _run_sequence(model, refs: list[dict], targets: list[dict], engine: str, use_multi_cache: bool, cache_by_ref: dict[str, dict] | None = None) -> None:
    out_dir = OUTPUT_ROOT / engine
    rows: list[ResultRow] = []
    sequence = [(ref, target) for target in targets for ref in refs]
    for ref, target in sequence:
        if use_multi_cache:
            _restore_reference_cache(model, cache_by_ref[ref["ref_audio_ascii"]])

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
                **INFER_KWARGS,
            )

        ok, seconds, duration, rtf, error = timed_generate(call, output_path)
        rows.append(
            ResultRow(
                engine,
                ref["voice_id"],
                ref["speaker"],
                phrase_id,
                text,
                1,
                ok,
                str(output_path) if ok else "",
                seconds,
                duration,
                rtf,
                error,
            )
        )
        write_manifest(rows, out_dir)
        print(f"{engine} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    refs = prepare_ascii_refs(OUTPUT_ROOT / "_indextts2_ref_cache_probe_refs")[:2]
    targets = [target for target in extract_target_texts() if target["phrase_id"] in TARGET_IDS]

    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=True,
    )

    warmup_path = OUTPUT_ROOT / "_indextts2_ref_cache_probe_warmup.wav"
    print("IndexTTS2 ref cache probe compile warmup start", flush=True)
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        **INFER_KWARGS,
    )
    print("IndexTTS2 ref cache probe compile warmup done", flush=True)

    _clear_reference_cache(model)
    _run_sequence(model, refs, targets, "IndexTTS2-cacheprobe-singlecache-alternating", use_multi_cache=False)

    print("IndexTTS2 ref cache probe precomputing reference caches", flush=True)
    cache_by_ref: dict[str, dict] = {}
    for ref in refs:
        _clear_reference_cache(model)
        prep_path = OUTPUT_ROOT / f"_indextts2_ref_cache_probe_prep_ref{ref['ref_index']:02d}.wav"
        model.infer(
            spk_audio_prompt=ref["ref_audio_ascii"],
            text=targets[0]["text"],
            output_path=str(prep_path),
            verbose=False,
            max_text_tokens_per_segment=120,
            **INFER_KWARGS,
        )
        cache_by_ref[ref["ref_audio_ascii"]] = _capture_reference_cache(model, ref["ref_audio_ascii"])
        print(f"cached reference {ref['speaker']}", flush=True)

    _clear_reference_cache(model)
    _run_sequence(model, refs, targets, "IndexTTS2-cacheprobe-multicache-alternating", use_multi_cache=True, cache_by_ref=cache_by_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
