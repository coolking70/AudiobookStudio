from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import requests

from common_japanese_sample_lines import (
    OUTPUT_ROOT,
    ResultRow,
    build_listen_page,
    extract_target_texts,
    load_references,
    wav_duration,
    write_manifest,
)


BASE_URL = "https://confucius4-tts.youdao.com/gradio"
ENGINE = "Confucius4-TTS-api"
OUT_DIR = OUTPUT_ROOT / ENGINE


def upload_reference(session: requests.Session, ref_path: Path) -> dict:
    with ref_path.open("rb") as f:
        response = session.post(
            f"{BASE_URL}/upload",
            files={"files": (ref_path.name, f, "audio/wav")},
            timeout=60,
        )
    response.raise_for_status()
    server_path = response.json()[0]
    return {
        "path": server_path,
        "orig_name": ref_path.name,
        "mime_type": "audio/wav",
        "meta": {"_type": "gradio.FileData"},
    }


def synthesize(session: requests.Session, text: str, ref_filedata: dict, output_path: Path) -> str:
    session_hash = str(uuid.uuid4())
    payload = {
        "data": [text, "zh", ref_filedata, None],
        "event_data": None,
        "fn_index": 1,
        "trigger_id": 9,
        "session_hash": session_hash,
    }
    joined = session.post(f"{BASE_URL}/queue/join", json=payload, timeout=60)
    joined.raise_for_status()

    with session.get(
        f"{BASE_URL}/queue/data",
        params={"session_hash": session_hash},
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("msg") != "process_completed":
                continue
            if not event.get("success", False):
                raise RuntimeError(json.dumps(event, ensure_ascii=False)[:1000])
            data = event.get("output", {}).get("data") or []
            if not data or not isinstance(data[0], dict):
                raise RuntimeError(f"Unexpected output: {data!r}")
            audio_url = data[0].get("url")
            status = data[1] if len(data) > 1 else ""
            if not audio_url:
                raise RuntimeError(f"Missing output audio URL: {data!r}")
            audio = session.get(audio_url, timeout=120)
            audio.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio.content)
            return str(status)
    raise RuntimeError("Queue stream ended before process_completed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs = load_references()
    targets = extract_target_texts()
    rows: list[ResultRow] = []
    session = requests.Session()

    uploaded_refs: dict[str, dict] = {}
    for ref in refs:
        ref_path = Path(ref["file"])
        uploaded_refs[ref["voice_id"]] = upload_reference(session, ref_path)

    for ref in refs:
        ref_index = int(ref.get("ref_index", len(uploaded_refs)))
        for target in targets:
            output_path = OUT_DIR / f"ref{ref_index:02d}__{target['phrase_id']}.wav"
            start = time.perf_counter()
            ok = False
            duration = 0.0
            rtf = 0.0
            error = ""
            try:
                if not output_path.exists():
                    synthesize(session, target["text"], uploaded_refs[ref["voice_id"]], output_path)
                seconds = time.perf_counter() - start
                duration = wav_duration(output_path)
                rtf = seconds / duration if duration > 0 else 0.0
                ok = True
            except Exception as exc:
                seconds = time.perf_counter() - start
                error = f"{exc.__class__.__name__}: {exc}"
            rows.append(
                ResultRow(
                    engine=ENGINE,
                    voice_id=ref["voice_id"],
                    speaker=ref["speaker"],
                    phrase_id=target["phrase_id"],
                    text=target["text"],
                    take=1,
                    ok=ok,
                    file=str(output_path),
                    seconds=seconds,
                    duration_seconds=duration,
                    rtf=rtf,
                    error=error,
                )
            )
            print(
                f"{ENGINE} {ref['speaker']} {target['phrase_id']} "
                f"ok={ok} seconds={seconds:.2f} rtf={rtf:.2f} {error}"
            )

    write_manifest(rows, OUT_DIR)
    page = build_listen_page(OUTPUT_ROOT)
    print(f"manifest: {OUT_DIR / 'manifest.json'}")
    print(f"listen: {page}")


if __name__ == "__main__":
    main()
