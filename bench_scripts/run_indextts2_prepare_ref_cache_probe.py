from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

from common_japanese_sample_lines import (
    OUTPUT_ROOT,
    ResultRow,
    extract_target_texts,
    prepare_ascii_refs,
    timed_generate,
    write_manifest,
)
from indextts2_cached_adapter import IndexTTS2CachedAdapter


INDEX_ROOT = Path(r"I:\code\aitts\index-tts")
MODEL_DIR = INDEX_ROOT / "checkpoints"
INFER_KWARGS = {
    "num_beams": 1,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 20,
}


def _seed_for(ref: dict, target: dict) -> int:
    value = f"{ref['voice_id']}::{target['phrase_id']}"
    seed = 20260613
    for char in value:
        seed = ((seed * 131) + ord(char)) % 2_147_483_647
    return seed


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_reference_cache(model) -> None:
    model.cache_spk_cond = None
    model.cache_s2mel_style = None
    model.cache_s2mel_prompt = None
    model.cache_mel = None
    model.cache_spk_audio_prompt = None
    model.cache_emo_cond = None
    model.cache_emo_audio_prompt = None


def _run_sequence(
    model,
    refs: list[dict],
    targets: list[dict],
    engine: str,
    cached_adapter: IndexTTS2CachedAdapter | None = None,
) -> None:
    out_dir = OUTPUT_ROOT / engine
    rows: list[ResultRow] = []
    sequence = [(ref, target) for target in targets for ref in refs]
    for ref, target in sequence:
        phrase_id = target["phrase_id"]
        text = target["text"]
        output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

        def call() -> None:
            _set_seed(_seed_for(ref, target))
            kwargs = dict(
                spk_audio_prompt=ref["ref_audio_ascii"],
                text=text,
                output_path=str(output_path),
                verbose=False,
                max_text_tokens_per_segment=120,
                **INFER_KWARGS,
            )
            if cached_adapter is not None:
                cached_adapter.infer_with_reference(ref["voice_id"], **kwargs)
            else:
                model.infer(**kwargs)

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

    refs = prepare_ascii_refs(OUTPUT_ROOT / "_indextts2_prepare_ref_cache_probe_refs")[:2]
    targets = extract_target_texts()

    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=True,
    )

    warmup_path = OUTPUT_ROOT / "_indextts2_prepare_ref_cache_probe_warmup.wav"
    print("IndexTTS2 prepare-ref cache probe compile warmup start", flush=True)
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        **INFER_KWARGS,
    )
    print("IndexTTS2 prepare-ref cache probe compile warmup done", flush=True)

    _clear_reference_cache(model)
    _run_sequence(model, refs, targets, "IndexTTS2-cacheprep-singlecache-alt12")

    print("IndexTTS2 prepare-ref cache probe precomputing reference caches", flush=True)
    adapter = IndexTTS2CachedAdapter(model)
    for ref in refs:
        prepared = adapter.prepare_reference(ref["voice_id"], ref["ref_audio_ascii"])
        print(f"prepared reference {ref['speaker']} sec={prepared.seconds:.2f}", flush=True)

    prep_rows = [
        {
            "ref_index": ref["ref_index"],
            "voice_id": ref["voice_id"],
            "speaker": ref["speaker"],
            "file": prepared.audio_path,
            "seconds": prepared.seconds,
        }
        for ref in refs
        for prepared in [adapter.references[ref["voice_id"]]]
    ]
    prep_path = OUTPUT_ROOT / "IndexTTS2-cacheprep-multicache-alt12" / "reference_prepare_times.json"
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    prep_path.write_text(json.dumps(prep_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    adapter.clear_active_reference()
    _run_sequence(model, refs, targets, "IndexTTS2-cacheprep-multicache-alt12", cached_adapter=adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
