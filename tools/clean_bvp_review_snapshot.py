from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
BOOK_VOICE_PARSER = ROOT / "BookVoiceParser"
if str(BOOK_VOICE_PARSER) not in sys.path:
    sys.path.insert(0, str(BOOK_VOICE_PARSER))

from book_voice_parser.alias_registry import AliasRegistry, clean_role_hint_name
from book_voice_parser.candidate_gen import generate_candidates
from book_voice_parser.schema import QuoteSpan


SKIP_SPEAKERS = {"", "旁白", "未知", "UNKNOWN", "众人"}
TEMPORARY_SPEAKERS = {
    "朋友",
    "朋友A",
    "朋友B",
    "朋友C",
    "少女",
    "女孩子",
    "旁边的孩子",
    "对面的少女",
    "访客",
    "金发女性",
    "另一个女孩",
    "未命名人物",
    "其他角色",
    "女性",
}
RELATION_SPEAKERS = {"妹妹", "姐姐", "弟弟", "哥哥", "妈妈", "母亲", "爸爸", "父亲"}
REVIEW_NOTE_RE = re.compile(r"(?:；|;)?LLM复核(?:待人工|失败|:|：|确认).*?(?=(?:；|;)[^；;]*$|$)")


def _is_bad_candidate(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.strip("\"'“”‘’：:").lower()
    if lowered in {"owner", "roles", "role", "aliases", "alias", "narrator"}:
        return True
    return bool(re.search(r"[{}\[\]]", text))


def _speaker_name(value: Any) -> str:
    name = clean_role_hint_name(value)
    if name in SKIP_SPEAKERS:
        return ""
    return name


def _infer_role_hints(segments: list[dict[str, Any]], min_count: int) -> dict[str, list[str]]:
    counts = Counter(_speaker_name(seg.get("speaker")) for seg in segments)
    names: set[str] = set()
    for name, count in counts.items():
        if not name or count < min_count:
            continue
        if name in TEMPORARY_SPEAKERS or name in RELATION_SPEAKERS:
            continue
        names.add(name)

    hints: dict[str, list[str]] = {}
    for name in sorted(names):
        aliases: set[str] = set()
        if len(name) >= 3:
            aliases.add(name[-2:])
        if len(name) >= 4:
            aliases.add(name[-3:])
        if name == "甘织玲奈子":
            aliases.update({"玲奈子", "玲奈亲", "小玲奈"})
        elif name == "小柳香穗":
            aliases.update({"香穗", "小香穗", "小柳同学"})
        elif name == "王冢真唯":
            aliases.update({"真唯", "小真唯"})
        elif name == "琴纱月":
            aliases.update({"纱月"})
        elif name == "甘织遥奈":
            aliases.update({"遥奈"})
        aliases.discard(name)
        hints[name] = sorted(alias for alias in aliases if alias)
    return hints


def _load_role_hints(path: Path | None, segments: list[dict[str, Any]], min_count: int) -> dict[str, Any] | list[str]:
    if path is None:
        return _infer_role_hints(segments, min_count)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _guess_narrator(segments: list[dict[str, Any]]) -> str | None:
    counts = Counter(_speaker_name(seg.get("speaker")) for seg in segments)
    for name, _ in counts.most_common():
        if name and name not in TEMPORARY_SPEAKERS and name not in RELATION_SPEAKERS:
            return name
    return None


def _clean_review_evidence(text: Any) -> str:
    cleaned = REVIEW_NOTE_RE.sub("", str(text or "")).strip("；; ")
    return cleaned


def _quote_from_segment(seg: dict[str, Any]) -> QuoteSpan:
    text = str(seg.get("text") or "")
    return QuoteSpan(
        quote_id=str(seg.get("quote_id") or ""),
        text=text,
        start=0,
        end=len(text),
        context_before=str(seg.get("context_before") or ""),
        context_after=str(seg.get("context_after") or ""),
    )


def clean_snapshot(
    snapshot: dict[str, Any],
    role_hints: dict[str, Any] | list[str],
    *,
    max_candidates: int,
    reset_review_evidence: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    output = deepcopy(snapshot)
    segments = output.get("segments") or []
    aliases = AliasRegistry.from_role_hints(role_hints)
    narrator = _guess_narrator(segments)
    recent_speakers: list[str] = []
    stats = Counter()

    for seg in segments:
        old_candidates = list(seg.get("candidates") or [])
        old_bad = any(_is_bad_candidate(item) for item in old_candidates)
        quote = _quote_from_segment(seg)
        cset = generate_candidates(
            quote,
            aliases=aliases,
            recent_speakers=recent_speakers,
            max_candidates=max_candidates,
            narrator=narrator,
        )

        seg["candidates"] = cset.candidates
        seg["candidate_sources"] = cset.candidate_sources
        seg["scene_characters"] = cset.scene_characters

        if old_bad:
            stats["segments_with_dirty_candidates_fixed"] += 1
        if seg.get("_needs_review"):
            stats["review_segments_cleaned"] += 1
            if reset_review_evidence:
                evidence = _clean_review_evidence(seg.get("evidence") or seg.get("_evidence") or "")
                if evidence:
                    evidence = f"{evidence}；已清理旧复核建议并重建候选池"
                else:
                    evidence = "已清理旧复核建议并重建候选池"
                seg["evidence"] = evidence
                seg["_evidence"] = evidence
            seg["_llm_skipped"] = False
            seg["_skip_reason"] = ""

        speaker = _speaker_name(seg.get("speaker"))
        if speaker and speaker not in SKIP_SPEAKERS:
            recent_speakers.append(speaker)

    output["snapshotCleanup"] = {
        "tool": "tools/clean_bvp_review_snapshot.py",
        "role_hint_count": len(aliases.known_names()),
        "narrator": narrator,
        "max_candidates": max_candidates,
        "reset_review_evidence": reset_review_evidence,
    }
    return output, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean BVP review snapshot candidates without rerunning full analysis.")
    parser.add_argument("input", type=Path, help="Task snapshot JSON to clean.")
    parser.add_argument("-o", "--output", type=Path, help="Output snapshot JSON path.")
    parser.add_argument("--role-hints", type=Path, help="Optional role hints JSON file.")
    parser.add_argument("--min-role-count", type=int, default=2, help="Minimum observed speaker count for inferred role hints.")
    parser.add_argument("--max-candidates", type=int, default=8, help="Maximum rebuilt candidates per segment.")
    parser.add_argument("--keep-review-evidence", action="store_true", help="Keep previous LLM review notes.")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("segments"), list):
        raise SystemExit("Input must be a task snapshot object with a segments array.")

    role_hints = _load_role_hints(args.role_hints, snapshot["segments"], args.min_role_count)
    cleaned, stats = clean_snapshot(
        snapshot,
        role_hints,
        max_candidates=max(3, args.max_candidates),
        reset_review_evidence=not args.keep_review_evidence,
    )
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_cleaned{args.input.suffix}")
    with output.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False, indent=2)

    print(json.dumps({"output": str(output), "stats": stats, "cleanup": cleaned.get("snapshotCleanup")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
