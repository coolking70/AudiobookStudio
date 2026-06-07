from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any


DEFAULT_BOOKMARKS = Path(r"E:\Temp\audiobook_bookmarks_export.json")
DEFAULT_LRC = Path(r"I:\code\aitts\omnivoice-reader\outputs\muli_4_parts\muli_4_part_001.lrc")
DEFAULT_RAW_SNAPSHOT = Path(r"C:\Users\coolking\Downloads\task_snapshot_segments_2026-06-05_1601.json")
DEFAULT_REVIEWED_SNAPSHOT = Path(
    r"C:\Users\coolking\Downloads\task_snapshot_segments_2026-06-05_1601_manual_reviewed_allrole_backcheck.json"
)
DEFAULT_OUT_DIR = Path(r"I:\code\aitts\omnivoice-reader\docs\samples")

TRACK_FILE = "muli_4_part_001"
SAMPLE_END_SECONDS = 60 * 60

# These are the bookmark-backed corrections we want to keep as a stable regression sample.
# `corrected_speaker` is the expected post-processing result.
BOOKMARK_EXPECTATIONS: dict[int, tuple[str, str]] = {
    60: ("甘织玲奈子", "后续旁白以一人称承接，当前台词不是紫阳花自述"),
    62: ("甘织玲奈子", "夹在主角内心独白中，属于玲奈子卡壳"),
    64: ("甘织玲奈子", "后续紫阳花接「……可？」说明前一句是玲奈子卡壳"),
    96: ("王冢真唯", "下一句旁白明确写「真唯的声音」"),
    123: ("王冢真唯", "前文明确为「真唯这样说道」"),
    133: ("甘织玲奈子", "后文说明这是玲奈子对真唯倾吐的心声"),
    169: ("甘织玲奈子", "第一人称自我怀疑，不能归给被提到的紫阳花"),
    171: ("甘织玲奈子", "旁白「我曾三番两次问过真唯为什么」被误切成台词"),
    204: ("甘织遥奈", "台词称呼「姐姐」，说话人应是妹妹遥奈"),
    219: ("甘织玲奈子", "前文写在脑海中反驳妹妹，是玲奈子独白"),
    224: ("甘织玲奈子", "妹妹闯入房间后，玲奈子对闯入者抱怨"),
    228: ("甘织遥奈", "台词称呼「姐姐」，说话人应是妹妹遥奈"),
    229: ("甘织玲奈子", "自称从出生到现在一直是垃圾，后文说明妹妹听后低沉回应"),
    230: ("甘织遥奈", "对玲奈子自嘲的反应"),
    233: ("甘织玲奈子", "对妹妹说「国中时代的姐姐才是垃圾」的反驳"),
    234: ("甘织遥奈", "承接上一句吐槽「不是你自己说的吗」"),
    249: ("甘织遥奈", "台词称呼「姐姐」，说话人应是妹妹遥奈"),
    250: ("甘织玲奈子", "回应上一句「我说姐姐」"),
    254: ("甘织玲奈子", "妹妹质问后，叙述者抬头并结巴辩解，后文才写妹妹翻白眼"),
    256: ("甘织遥奈", "称呼「姐姐」且内容评价姐姐翘课"),
    259: ("甘织玲奈子", "后文「我偷偷观察遥奈」说明这是玲奈子的反应"),
    271: ("甘织玲奈子", "主角抗辩自己没有翘课"),
    278: ("甘织遥奈", "妹妹拿手机回复后卖弄恩情，称呼姐姐"),
    292: ("甘织遥奈", "后文明确写妹妹行礼后离开"),
    312: ("长谷川同学", "前文长谷川和平野走来，小香穗尚未登场"),
    313: ("平野同学", "另一位普通同学询问身体状况"),
    317: ("长谷川同学", "普通同学承接身体状况话题"),
    318: ("平野同学", "普通同学夸奖班级华丽度，不应归小柳香穗"),
    319: ("甘织玲奈子", "后文「我这样说完，两人也捧场地笑」"),
    350: ("琴纱月", "紫阳花上一句已问候小玲奈，称呼「甘织」更符合纱月"),
}


def parse_lrc_timestamp(value: str) -> float:
    match = re.match(r"(\d+):(\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Invalid LRC timestamp: {value}")
    return int(match.group(1)) * 60 + float(match.group(2))


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_lrc(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\[(\d+:\d+(?:\.\d+)?)\](.*)", raw)
        if not match:
            continue
        start = parse_lrc_timestamp(match.group(1))
        if start <= SAMPLE_END_SECONDS:
            entries.append({"start": start, "time": format_timestamp(start), "text": match.group(2)})
    return entries


def load_manual_bookmarks(path: Path, track_file: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for section in data.get("sections") or []:
        if section.get("fileName") == track_file:
            return sorted(section.get("manualBookmarks") or [], key=lambda item: item.get("positionMs") or 0)
    return []


def clean_segment(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker": raw.get("speaker"),
        "text": raw.get("text"),
        "quote_id": raw.get("quote_id"),
        "confidence": raw.get("confidence"),
        "evidence": raw.get("evidence"),
        "attribution_type": raw.get("attribution_type"),
        "candidates": raw.get("candidates") or [],
        "scene_characters": raw.get("scene_characters") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bookmark-backed role attribution regression sample.")
    parser.add_argument("--bookmarks", type=Path, default=DEFAULT_BOOKMARKS)
    parser.add_argument("--lrc", type=Path, default=DEFAULT_LRC)
    parser.add_argument("--raw-snapshot", type=Path, default=DEFAULT_RAW_SNAPSHOT)
    parser.add_argument("--reviewed-snapshot", type=Path, default=DEFAULT_REVIEWED_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    lrc_entries = load_lrc(args.lrc)
    starts = [item["start"] for item in lrc_entries]
    bookmarks = load_manual_bookmarks(args.bookmarks, TRACK_FILE)
    raw_segments = json.loads(args.raw_snapshot.read_text(encoding="utf-8"))["segments"]
    reviewed_segments = json.loads(args.reviewed_snapshot.read_text(encoding="utf-8"))["segments"]

    bookmark_rows = []
    for bookmark in bookmarks:
        position = float(bookmark.get("positionMs") or 0) / 1000.0
        if position > SAMPLE_END_SECONDS:
            continue
        idx = bisect_right(starts, position) - 1
        context = lrc_entries[max(0, idx - 2) : min(len(lrc_entries), idx + 3)]
        bookmark_rows.append(
            {
                "time": format_timestamp(position),
                "position_seconds": round(position, 3),
                "label": bookmark.get("label"),
                "matched_lrc_index": idx,
                "matched_lrc": lrc_entries[idx] if 0 <= idx < len(lrc_entries) else None,
                "context": context,
            }
        )

    corrected_segments = []
    for index, (expected, reason) in BOOKMARK_EXPECTATIONS.items():
        raw = clean_segment(raw_segments[index])
        reviewed = clean_segment(reviewed_segments[index])
        corrected_segments.append(
            {
                "index": index,
                "raw_speaker": raw["speaker"],
                "reviewed_speaker": reviewed["speaker"],
                "expected_speaker": expected,
                "reason": reason,
                "raw": raw,
                "reviewed": reviewed,
            }
        )

    sample_text = "\n".join(item["text"] for item in lrc_entries).strip() + "\n"
    payload = {
        "source": {
            "track_file": TRACK_FILE,
            "bookmarks": str(args.bookmarks),
            "lrc": str(args.lrc),
            "raw_snapshot": str(args.raw_snapshot),
            "reviewed_snapshot": str(args.reviewed_snapshot),
            "sample_end_seconds": SAMPLE_END_SECONDS,
        },
        "sample_text": sample_text,
        "manual_bookmarks": bookmark_rows,
        "corrected_segments": corrected_segments,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "muli4_part001_first_hour_bookmark_regression.json"
    text_path = args.out_dir / "muli4_part001_first_hour_sample.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(sample_text, encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {text_path}")
    print(f"bookmarks={len(bookmark_rows)} corrected_segments={len(corrected_segments)} lrc_entries={len(lrc_entries)}")


if __name__ == "__main__":
    main()
