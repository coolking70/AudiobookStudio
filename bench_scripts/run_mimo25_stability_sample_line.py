from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = "mimo-v2.5-tts-voiceclone"
TAKES = 5
PCM_SAMPLE_RATE = 24000


def _voice_data_url(ref_audio_path: str) -> str:
    encoded = base64.b64encode(Path(ref_audio_path).read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _post_json(payload: dict, api_key: str) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo API HTTP {exc.code}: {detail}") from exc


def _stream_json(payload: dict, api_key: str) -> Iterator[dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo API HTTP {exc.code}: {detail}") from exc


def _extract_nonstream_audio(response: dict) -> bytes:
    try:
        data = response["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"MiMo API response did not contain audio data: {response}") from exc
    return base64.b64decode(data)


def _extract_stream_audio_chunks(events: Iterator[dict]) -> bytes:
    chunks: list[bytes] = []
    for event in events:
        try:
            data = event["choices"][0]["delta"]["audio"]["data"]
        except (KeyError, IndexError, TypeError):
            continue
        if data:
            chunks.append(base64.b64decode(data))
    if not chunks:
        raise RuntimeError("MiMo streaming response did not contain audio chunks")
    return b"".join(chunks)


def _write_pcm16_wav(output_path: Path, pcm_bytes: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(PCM_SAMPLE_RATE)
        wav.writeframes(pcm_bytes)


def _payload(text: str, voice: str, *, stream: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": text},
        ],
        "audio": {
            "format": "pcm16" if stream else "wav",
            "voice": voice,
        },
    }
    if stream:
        payload["stream"] = True
    return payload


def _run_engine(engine: str, *, stream: bool, api_key: str, ref: dict, targets: list[dict], voice: str) -> None:
    out_dir = OUTPUT_ROOT / engine
    rows: list[ResultRow] = []
    for take, target in enumerate(targets[:TAKES], start=1):
        phrase_id = target["phrase_id"]
        text = target["text"]
        output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}__take{take:02d}.wav"

        def call() -> None:
            payload = _payload(text, voice, stream=stream)
            if stream:
                _write_pcm16_wav(output_path, _extract_stream_audio_chunks(_stream_json(payload, api_key)))
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(_extract_nonstream_audio(_post_json(payload, api_key)))

        ok, seconds, duration, rtf, error = timed_generate(call, output_path)
        rows.append(ResultRow(engine, ref["voice_id"], ref["speaker"], phrase_id, text, take, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
        write_manifest(rows, out_dir)
        print(f"{engine} take{take:02d} ok={ok} sec={seconds:.2f}", flush=True)


def main() -> int:
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise RuntimeError("MIMO_API_KEY is required")
    refs = prepare_ascii_refs(OUTPUT_ROOT / "MiMo-v2.5-stability_refs")
    targets = extract_target_texts()
    ref = refs[0]
    voice = _voice_data_url(ref["ref_audio_ascii"])
    _run_engine("MiMo-v2.5-nonstream-varied-stability", stream=False, api_key=api_key, ref=ref, targets=targets, voice=voice)
    _run_engine("MiMo-v2.5-stream-varied-stability", stream=True, api_key=api_key, ref=ref, targets=targets, voice=voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
