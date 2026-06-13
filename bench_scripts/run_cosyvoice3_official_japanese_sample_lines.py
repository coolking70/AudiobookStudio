from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torchaudio

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


COSYVOICE_ROOT = Path(r"I:\code\aitts\CosyVoice")
MODEL_PATH = COSYVOICE_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", action="store_true", help="Use llm.rl.pt instead of llm.pt.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(COSYVOICE_ROOT))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    engine = "CosyVoice3-0.5B-RL-official-format" if args.rl else "CosyVoice3-0.5B-official-format"
    out_dir = OUTPUT_ROOT / engine
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    model = CosyVoice3(str(MODEL_PATH), load_trt=False, load_vllm=False, fp16=True)
    if args.rl:
        model.model.load(str(MODEL_PATH / "llm.rl.pt"), str(MODEL_PATH / "flow.pt"), str(MODEL_PATH / "hift.pt"))

    rows: list[ResultRow] = []
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                # CosyVoice3 expects the assistant prompt before <|endofprompt|>, then the TTS text.
                official_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                items = list(model.inference_cross_lingual(official_text, ref["ref_audio_ascii"], stream=False))
                if not items:
                    raise RuntimeError("CosyVoice3 returned no output")
                torchaudio.save(str(output_path), items[0]["tts_speech"].cpu(), model.sample_rate)

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(engine, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{engine} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
