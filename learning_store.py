"""Local append-only storage for human speaker corrections (silver data)."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any

from output_layout import OUTPUT_DIR

_LOCK = threading.Lock()
SILVER_DIR = OUTPUT_DIR / "learning"
SILVER_PATH = SILVER_DIR / "silver.jsonl"


def _digest(text: str, segment: dict[str, Any], index: int, speaker: str) -> str:
    raw = json.dumps({"text": text, "index": index, "segment": segment.get("text", ""), "speaker": speaker}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_correction(
    *, text: str, segment: dict[str, Any], index: int, previous_speaker: str, source: str = "ui"
) -> dict[str, Any]:
    speaker = str(segment.get("speaker") or "").strip()
    previous = str(previous_speaker or "").strip()
    if not speaker or speaker == previous or not str(segment.get("text") or "").strip():
        return {"stored": False, "reason": "unchanged"}
    record = {
        "version": 1,
        "id": _digest(text, segment, index, speaker),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "text_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
        "segment_index": int(index),
        "text": str(segment.get("text") or ""),
        "context_before": str(segment.get("context_before") or "")[-500:],
        "context_after": str(segment.get("context_after") or "")[:500],
        "previous_speaker": previous,
        "speaker": speaker,
        "confidence": segment.get("confidence"),
        "evidence": str(segment.get("evidence") or segment.get("_evidence") or "")[-1000:],
    }
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        existing = set()
        if SILVER_PATH.exists():
            for line in SILVER_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    existing.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue
        if record["id"] in existing:
            return {"stored": False, "reason": "duplicate", "id": record["id"]}
        with SILVER_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"stored": True, "id": record["id"], "record": record}


def read_records() -> list[dict[str, Any]]:
    if not SILVER_PATH.exists():
        return []
    records = []
    for line in SILVER_PATH.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("version") == 1:
            records.append(value)
    return records


def stats() -> dict[str, Any]:
    records = read_records()
    speakers: dict[str, int] = {}
    for record in records:
        speaker = str(record.get("speaker") or "")
        speakers[speaker] = speakers.get(speaker, 0) + 1
    return {
        "count": len(records),
        "speakers": dict(sorted(speakers.items())),
        "path": str(SILVER_PATH),
        "quality": audit_records(records),
    }


def audit_records(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Validate silver records before they can be promoted into a dataset version."""
    values = list(read_records() if records is None else records)
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(values):
        record_id = str(record.get("id") or "")
        if not record_id:
            issues.append({"index": index, "code": "missing_id"})
        elif record_id in seen_ids:
            issues.append({"index": index, "code": "duplicate_id"})
        seen_ids.add(record_id)
        if not str(record.get("text") or "").strip():
            issues.append({"index": index, "code": "empty_text"})
        speaker = str(record.get("speaker") or "").strip()
        previous = str(record.get("previous_speaker") or "").strip()
        if not speaker:
            issues.append({"index": index, "code": "missing_speaker"})
        if speaker and speaker == previous:
            issues.append({"index": index, "code": "unchanged_speaker"})
        if not str(record.get("text_sha256") or ""):
            issues.append({"index": index, "code": "missing_source_hash"})
    return {
        "records": len(values),
        "valid": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues[:100],
    }


def build_version_bundle(
    records: list[dict[str, Any]] | None = None,
    *,
    min_records: int = 20,
) -> dict[str, Any]:
    """Build an immutable, checksummed training bundle after quality gates pass."""
    values = list(read_records() if records is None else records)
    quality = audit_records(values)
    if not quality["valid"]:
        raise ValueError(f"silver quality gate failed with {quality['issue_count']} issue(s)")
    if len(values) < max(1, int(min_records)):
        raise ValueError(f"silver dataset requires at least {max(1, int(min_records))} records")
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "dataset_id": f"silver-{digest[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": values,
        "record_count": len(values),
        "sha256": digest,
        "quality": quality,
    }
