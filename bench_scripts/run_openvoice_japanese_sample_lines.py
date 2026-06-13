from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "OpenVoiceV2"
OPENVOICE_ROOT = Path(r"I:\code\aitts\OpenVoice")
CHECKPOINTS = OPENVOICE_ROOT / "checkpoints_v2"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("NLTK_DATA", r"I:\hf_cache\nltk_data")
    sys.path.insert(0, str(OPENVOICE_ROOT))
    from melo.api import TTS
    from openvoice.api import ToneColorConverter

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = OUTPUT_ROOT / ENGINE
    base_dir = out_dir / "_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()

    tts = TTS(language="ZH", device=device)
    speaker_id = tts.hps.data.spk2id["ZH"]
    converter = ToneColorConverter(str(CHECKPOINTS / "converter" / "config.json"), device=device)
    converter.watermark_model = None
    converter.load_ckpt(str(CHECKPOINTS / "converter" / "checkpoint.pth"))
    source_se = torch.load(str(CHECKPOINTS / "base_speakers" / "ses" / "zh.pth"), map_location=device)

    rows: list[ResultRow] = []
    target_se_cache: dict[str, torch.Tensor] = {}
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            base_path = base_dir / f"ref{ref['ref_index']:02d}__{phrase_id}_base.wav"
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                if ref["ref_audio_ascii"] not in target_se_cache:
                    target_se_cache[ref["ref_audio_ascii"]] = converter.extract_se(ref["ref_audio_ascii"])
                tts.tts_to_file(text, speaker_id, str(base_path), speed=1.0)
                converter.convert(
                    str(base_path),
                    src_se=source_se,
                    tgt_se=target_se_cache[ref["ref_audio_ascii"]],
                    output_path=str(output_path),
                    message="sample-lines",
                )

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
