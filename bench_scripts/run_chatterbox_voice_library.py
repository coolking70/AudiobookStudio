from __future__ import annotations

from pathlib import Path

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

from common_voice_library import DEFAULT_OUTPUT_ROOT, PHRASES, REFERENCES, ResultRow, timed_generate, write_manifest


ENGINE = "Chatterbox-0.1.7"


def main() -> int:
    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")

    rows: list[ResultRow] = []
    for ref in REFERENCES:
        for phrase_id, text in PHRASES:
            for take in range(1, 3):
                output_path = out_dir / f"{ref['voice_id']}__{phrase_id}__take{take:02d}.wav"

                def call() -> None:
                    wav = model.generate(
                        text,
                        audio_prompt_path=ref["ref_audio"],
                        exaggeration=0.7,
                        temperature=0.8,
                    )
                    ta.save(str(output_path), wav, model.sr)

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
