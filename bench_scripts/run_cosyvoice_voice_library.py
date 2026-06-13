from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torchaudio

from common_voice_library import (
    DEFAULT_OUTPUT_ROOT,
    PHRASES,
    REFERENCES,
    ResultRow,
    timed_generate,
    write_manifest,
)


ENGINE = "CosyVoice-300M"
COSYVOICE_ROOT = Path(r"I:\code\aitts\CosyVoice")
MODEL_PATH = COSYVOICE_ROOT / "pretrained_models" / "CosyVoice-300M"


def safe_name(index: int) -> str:
    return f"ref{index:02d}"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(COSYVOICE_ROOT))

    from cosyvoice.cli.cosyvoice import AutoModel

    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = out_dir / "_ascii_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    ref_paths: dict[str, Path] = {}
    for index, ref in enumerate(REFERENCES, start=1):
        src = Path(ref["ref_audio"])
        dst = ref_dir / f"{safe_name(index)}{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        ref_paths[ref["voice_id"]] = dst

    model = AutoModel(model_dir=str(MODEL_PATH))

    rows: list[ResultRow] = []
    for ref in REFERENCES:
        prompt_wav = str(ref_paths[ref["voice_id"]])
        prompt_text = ref["ref_text"]
        for phrase_id, text in PHRASES:
            for take in range(1, 3):
                output_path = out_dir / f"{safe_name(REFERENCES.index(ref) + 1)}__{phrase_id}__take{take:02d}.wav"

                def call() -> None:
                    items = list(
                        model.inference_zero_shot(
                            text,
                            prompt_text,
                            prompt_wav,
                            stream=False,
                            text_frontend=False,
                        )
                    )
                    if not items:
                        raise RuntimeError("CosyVoice returned no output")
                    torchaudio.save(str(output_path), items[0]["tts_speech"].cpu(), model.sample_rate)

                ok, seconds, duration, rtf, error = timed_generate(call, output_path)
                rows.append(
                    ResultRow(
                        engine=ENGINE,
                        voice_id=ref["voice_id"],
                        speaker=ref["speaker"],
                        phrase_id=phrase_id,
                        text=text,
                        take=take,
                        ok=ok,
                        file=str(output_path) if ok else "",
                        seconds=seconds,
                        duration_seconds=duration,
                        rtf=rtf,
                        error=error,
                    )
                )
                print(f"{ENGINE} {ref['voice_id']} {phrase_id} take{take:02d} ok={ok} sec={seconds:.2f}", flush=True)

    write_manifest(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
