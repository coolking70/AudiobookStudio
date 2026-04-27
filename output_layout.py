from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = Path("outputs")
TEMP_BY_DATE_DIRNAME = "temp_by_date"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _sanitize_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def current_output_date(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime("%Y-%m-%d")


def get_temp_archive_root(when: datetime | None = None) -> Path:
    root = ensure_output_dir() / TEMP_BY_DATE_DIRNAME / current_output_date(when)
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_temp_archive_dir(category: str | None = None, when: datetime | None = None) -> Path:
    base = get_temp_archive_root(when)
    if category:
        base = base / _sanitize_component(category, "misc")
        base.mkdir(parents=True, exist_ok=True)
    return base


def build_temp_archive_path(filename: str, category: str | None = None, when: datetime | None = None) -> Path:
    return get_temp_archive_dir(category=category, when=when) / Path(filename).name


def is_temporary_output_name(name: str | None) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized.startswith("preview_")


def resolve_named_output_path(
    output_name: str,
    suffix: str,
    *,
    temp_category: str | None = None,
    when: datetime | None = None,
) -> Path:
    filename = f"{output_name}{suffix}"
    if is_temporary_output_name(output_name):
        return build_temp_archive_path(filename, category=temp_category or "preview", when=when)
    return ensure_output_dir() / filename


def get_segment_output_dir(base_name: str, when: datetime | None = None) -> Path:
    safe_base_name = _sanitize_component(base_name, "segments")
    out_dir = get_temp_archive_dir("segments", when=when) / safe_base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_segment_output_path(base_name: str, index: int, when: datetime | None = None) -> Path:
    safe_base_name = _sanitize_component(base_name, "segments")
    return get_segment_output_dir(safe_base_name, when=when) / f"{safe_base_name}_{index:03d}.wav"


def build_segment_output_paths(base_name: str, segment_count: int, when: datetime | None = None) -> list[Path]:
    return [build_segment_output_path(base_name, idx, when=when) for idx in range(1, segment_count + 1)]
