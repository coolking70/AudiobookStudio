from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "GPT-SoVITS-v2-empty-reftext"
GPT_ROOT = Path(r"I:\code\aitts\GPT-SoVITS")
CONFIG_PATH = GPT_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    cwd = Path.cwd()
    os.chdir(GPT_ROOT)
    sys.path.insert(0, str(GPT_ROOT))
    sys.path.insert(0, str(GPT_ROOT / "GPT_SoVITS"))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    if os.environ.get("AITTS_PROBE_ONLY") == "1":
        refs = refs[:1]
        targets = targets[:1]
    model = TTS(TTS_Config(str(CONFIG_PATH)))

    rows: list[ResultRow] = []
    try:
        for ref in refs:
            for target in targets:
                phrase_id = target["phrase_id"]
                text = target["text"]
                output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

                def call() -> None:
                    req = {
                        "text": text,
                        "text_lang": "zh",
                        "ref_audio_path": ref["ref_audio_ascii"],
                        "aux_ref_audio_paths": [],
                        "prompt_text": "",
                        "prompt_lang": "ja",
                        "top_k": 15,
                        "top_p": 1,
                        "temperature": 1,
                        "text_split_method": "cut5",
                        "batch_size": 1,
                        "batch_threshold": 0.75,
                        "split_bucket": True,
                        "speed_factor": 1.0,
                        "fragment_interval": 0.3,
                        "seed": -1,
                        "media_type": "wav",
                        "streaming_mode": False,
                        "parallel_infer": True,
                        "repetition_penalty": 1.35,
                        "sample_steps": 32,
                        "super_sampling": False,
                    }
                    sr, audio = next(model.run(req))
                    sf.write(str(output_path), audio, sr)

                ok, seconds, duration, rtf, error = timed_generate(call, output_path)
                rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
                write_manifest(rows, out_dir)
                print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    finally:
        os.chdir(cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
