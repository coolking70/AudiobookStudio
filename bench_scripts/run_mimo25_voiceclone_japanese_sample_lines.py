from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "MiMo-v2.5-TTS-voiceclone"
API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-tts-voiceclone"


def _post_json(payload: dict, api_key: str) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo API HTTP {exc.code}: {detail}") from exc


def _extract_audio_bytes(response: dict) -> bytes:
    try:
        data = response["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"MiMo API response did not contain audio data: {response}") from exc
    return base64.b64decode(data)


def _voice_data_url(ref_audio_path: str) -> str:
    encoded = base64.b64encode(Path(ref_audio_path).read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def main() -> int:
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise RuntimeError("MIMO_API_KEY is required")

    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    if os.environ.get("AITTS_PROBE_ONLY") == "1":
        refs = refs[:1]
        targets = targets[:1]

    rows: list[ResultRow] = []
    voice_cache: dict[str, str] = {}
    for ref in refs:
        voice = voice_cache.setdefault(ref["ref_audio_ascii"], _voice_data_url(ref["ref_audio_ascii"]))
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": text},
                    ],
                    "audio": {
                        "format": "wav",
                        "voice": voice,
                    },
                }
                audio_bytes = _extract_audio_bytes(_post_json(payload, api_key))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_bytes)

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
