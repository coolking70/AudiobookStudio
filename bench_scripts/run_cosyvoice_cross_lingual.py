from __future__ import annotations

import os
import sys
from pathlib import Path

import torchaudio

from common_cross_lingual import OUTPUT_ROOT, PHRASES, ResultRow, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "CosyVoice-300M"
COSYVOICE_ROOT = Path(r"I:\code\aitts\CosyVoice")
MODEL_PATH = COSYVOICE_ROOT / "pretrained_models" / "CosyVoice-300M"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(COSYVOICE_ROOT))

    from cosyvoice.cli.cosyvoice import AutoModel

    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    model = AutoModel(model_dir=str(MODEL_PATH))

    rows: list[ResultRow] = []
    for ref in refs:
        for phrase_id, text in PHRASES:
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                items = list(
                    model.inference_zero_shot(
                        text,
                        ref["ref_text"],
                        ref["ref_audio_ascii"],
                        stream=False,
                        text_frontend=False,
                    )
                )
                if not items:
                    raise RuntimeError("CosyVoice returned no output")
                torchaudio.save(str(output_path), items[0]["tts_speech"].cpu(), model.sample_rate)

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
