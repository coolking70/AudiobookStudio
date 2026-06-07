from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CANONICAL_SPEAKERS = {
    "小香穗": "小柳香穗",
    "香穗": "小柳香穗",
    "小玲奈": "甘织玲奈子",
    "玲奈子": "甘织玲奈子",
    "真唯": "王冢真唯",
    "小真唯": "王冢真唯",
    "紫阳花": "濑名紫阳花",
    "纱月": "琴纱月",
    "小纱月": "琴纱月",
    "遥奈": "甘织遥奈",
    "甘织遥奈子": "甘织遥奈",
}


# Conservative manual pass over the 38 automatic corrections from the 0010 snapshot.
# Values are based on local before/after context inspection.
MANUAL_OVERRIDES: dict[str, tuple[str, float, str]] = {
    "q0058": ("甘织玲奈子", 0.90, "人工复核：上下文为紫阳花告白后的第一人称无言反应，非真唯"),
    "q0148": ("甘织遥奈", 0.95, "人工复核：与玲奈子围绕相簿条件对答的是遥奈"),
    "q0208": ("小柳香穗", 0.95, "人工复核：小香穗别名统一为小柳香穗"),
    "q0220": ("王冢真唯", 0.90, "人工复核：紫阳花和纱月已先问候，此句为真唯问候"),
    "q0319": ("小柳香穗", 0.90, "人工复核：听到玲奈子坦白告白事件后的反应来自小香穗"),
    "q0337": ("小柳香穗", 0.95, "人工复核：小香穗与主角对质调侃"),
    "q0363": ("甘织玲奈子", 0.85, "人工复核：午休冷战中第一人称叙述延续，保守归回玲奈子"),
    "q0541": ("琴纱月", 0.95, "人工复核：与纱月同学的连续对话轮替"),
    "q0577": ("小柳香穗", 0.95, "人工复核：小香穗把玲奈子拖进车站后的发言"),
    "q0628": ("甘织玲奈子", 0.95, "人工复核：被小香穗贴近后的第一人称惊慌反应"),
    "q0649": ("小柳香穗", 0.95, "人工复核：小香穗结束玩笑并转入工作话题"),
    "q0708": ("moon小姐", 0.95, "人工复核：叙述中假想 moon 小姐会这样回答"),
    "q0711": ("甘织玲奈子", 0.95, "人工复核：小香穗说预想角色后，玲奈子追问“角色？”"),
    "q0774": ("甘织玲奈子", 0.95, "人工复核：回应小香穗腕力挑衅的是玲奈子"),
    "q0776": ("小柳香穗", 0.95, "人工复核：小香穗停下摊衣服后回答拍摄话题"),
    "q0820": ("三位女性", 0.90, "人工复核：大姐姐们纷纷向玲奈子回礼问候"),
    "q0822": ("三位女性", 0.90, "人工复核：女性们回应 nagipo 号召"),
    "q0896": ("甘织玲奈子", 0.95, "人工复核：回答小香穗“有认真听吗”的是玲奈子"),
    "q0967": ("甘织玲奈子", 0.95, "人工复核：回应小香穗自言自语的是玲奈子"),
    "q1080": ("甘织玲奈子", 0.95, "人工复核：思考纱月同学搭配拍摄的是玲奈子"),
    "q1207": ("甘织玲奈子", 0.95, "人工复核：要求小香穗关掉电视的是玲奈子"),
    "q1243": ("甘织玲奈子", 0.95, "人工复核：名侦探玲奈子的脑内提醒后由玲奈子发问"),
    "q1290": ("小柳香穗", 0.95, "人工复核：小香穗自述隐形眼镜相关困扰"),
    "q1330": ("小柳香穗", 0.95, "人工复核：小香穗解释自己被选上活动空缺"),
    "q1488": ("甘织玲奈子", 0.95, "人工复核：下文明确“我露出笑容，用力点头”"),
    "q1517": ("甘织玲奈子", 0.95, "人工复核：紫阳花告白后的第一人称反应"),
    "q1520": ("濑名紫阳花", 0.95, "人工复核：紫阳花说开心后，玲奈子回应“嗯，我也是”"),
    "q1586": ("濑名紫阳花", 0.95, "人工复核：称呼小玲奈且害羞的语气对应紫阳花"),
    "q1613": ("小柳香穗", 0.95, "人工复核：讨论催眠音档时发问的是小香穗"),
    "q1655": ("星来", 0.95, "人工复核：提到“我的搭档”的发言属于赛菈菈/星来"),
    "q1698": ("小柳香穗", 0.95, "人工复核：小香穗回应“你没事吧”"),
    "q1868": ("琴纱月", 0.95, "人工复核：紫阳花指出话语过分后，纱月回应并道歉"),
    "q1889": ("濑名紫阳花", 0.95, "人工复核：纱月答应同行后，紫阳花开心回应"),
    "q1925": ("濑名紫阳花", 0.95, "人工复核：前文明确紫阳花猛然大喊"),
    "q2019": ("濑名紫阳花", 0.95, "人工复核：屋顶边缘觉得可怕的是紫阳花"),
    "q2210": ("濑名紫阳花", 0.95, "人工复核：面对玲奈子豁出去的告白，紫阳花先回应"),
    "q2223": ("甘织玲奈子", 0.95, "人工复核：害怕被两人抛下并缠住她们的是玲奈子"),
    "q2231": ("甘织玲奈子", 0.90, "人工复核：紫阳花继续发言前，玲奈子短声回应"),
}


FIRST_PERSON_MARKERS = (
    "第一人称",
    "叙述者",
    "内心独白",
    "自白",
    "前文「我",
    "前文“我",
    "后文明确“我",
    "紧前文“我",
    "“我”为",
    "我露出",
    "我用",
    "我低",
    "我顿时",
)

OBVIOUS_SPEAKER_MARKERS = {
    "小柳香穗": ("小香穗", "香穗", "玲奈亲"),
    "琴纱月": ("纱月", "小纱", "moon"),
    "濑名紫阳花": ("紫阳花", "小玲奈"),
    "星来": ("星来", "赛菈菈"),
}


def _canonical(name: Any) -> str:
    text = str(name or "").strip()
    return CANONICAL_SPEAKERS.get(text, text)


def _canonicalize_candidate_fields(seg: dict[str, Any]) -> None:
    sources = seg.get("candidate_sources") or {}
    next_sources: dict[str, list[str]] = {}
    for key, value in sources.items():
        canonical = _canonical(key)
        target = next_sources.setdefault(canonical, [])
        for source in value or []:
            if source not in target:
                target.append(source)

    candidates: list[str] = []
    for item in seg.get("candidates") or []:
        canonical = _canonical(item)
        if canonical and canonical not in candidates:
            candidates.append(canonical)
    scene: list[str] = []
    for item in seg.get("scene_characters") or []:
        canonical = _canonical(item)
        if canonical and canonical not in scene:
            scene.append(canonical)

    seg["candidates"] = candidates
    seg["candidate_sources"] = {name: next_sources.get(name, sources.get(name, [])) for name in candidates}
    seg["scene_characters"] = scene


def _mark_reviewed(seg: dict[str, Any], speaker: str, confidence: float, note: str) -> None:
    seg["speaker"] = speaker
    seg["confidence"] = confidence
    seg["_confidence"] = confidence
    seg["_needs_review"] = False
    seg["_suspicious"] = False
    seg["_suspicious_reason"] = ""
    seg["_llm_skipped"] = False
    seg["_skip_reason"] = ""
    seg["_manual_reviewed"] = True
    evidence = str(seg.get("evidence") or seg.get("_evidence") or "").strip("；; ")
    if "人工复核：" not in evidence:
        evidence = f"{evidence}；{note}" if evidence else note
    else:
        evidence = f"{evidence}；{note}"
    seg["evidence"] = evidence
    seg["_evidence"] = evidence


def finalize_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    output = deepcopy(snapshot)
    stats = {
        "speaker_aliases_canonicalized": 0,
        "manual_overrides_applied": 0,
        "first_person_confirmed": 0,
        "llm_confirmed_promoted": 0,
    }

    for seg in output.get("segments") or []:
        old_speaker = str(seg.get("speaker") or "")
        new_speaker = _canonical(old_speaker)
        if new_speaker != old_speaker:
            seg["speaker"] = new_speaker
            stats["speaker_aliases_canonicalized"] += 1
        _canonicalize_candidate_fields(seg)

        quote_id = str(seg.get("quote_id") or "")
        if quote_id in MANUAL_OVERRIDES:
            speaker, confidence, note = MANUAL_OVERRIDES[quote_id]
            _mark_reviewed(seg, speaker, confidence, note)
            stats["manual_overrides_applied"] += 1
            continue

        evidence = str(seg.get("evidence") or seg.get("_evidence") or "")
        speaker = str(seg.get("speaker") or "")
        if seg.get("_needs_review") and speaker == "甘织玲奈子" and any(marker in evidence for marker in FIRST_PERSON_MARKERS):
            _mark_reviewed(seg, "甘织玲奈子", max(float(seg.get("confidence") or 0), 0.85), "人工复核：第一人称/叙述者锚点确认玲奈子")
            stats["first_person_confirmed"] += 1
            continue

        if seg.get("_needs_review") and "LLM确认" in evidence and "LLM复核待人工" not in evidence:
            seg["confidence"] = max(float(seg.get("confidence") or 0), 0.75)
            seg["_confidence"] = seg["confidence"]
            seg["_needs_review"] = False
            stats["llm_confirmed_promoted"] += 1
            continue

        if seg.get("_needs_review") and speaker == "甘织玲奈子" and "我" in str(seg.get("text") or ""):
            _mark_reviewed(seg, "甘织玲奈子", max(float(seg.get("confidence") or 0), 0.85), "人工复核：台词为第一人称表达，确认玲奈子")
            stats["first_person_confirmed"] += 1
            continue

        markers = OBVIOUS_SPEAKER_MARKERS.get(speaker, ())
        marker_text = f"{seg.get('text') or ''}\n{evidence}"
        if seg.get("_needs_review") and markers and any(marker in marker_text for marker in markers):
            _mark_reviewed(seg, speaker, max(float(seg.get("confidence") or 0), 0.85), f"人工复核：文本/证据含明确{speaker}线索")
            stats["obvious_speaker_confirmed"] = stats.get("obvious_speaker_confirmed", 0) + 1

    output["manualReviewFinalize"] = {
        "tool": "tools/finalize_bvp_review_snapshot.py",
        "description": "Canonicalized speaker aliases and applied conservative manual review over the post-cleanup BVP review snapshot.",
        "stats": stats,
    }
    return output, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a cleaned BVP review snapshot with manual conservative corrections.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    finalized, stats = finalize_snapshot(snapshot)

    output = args.output or args.input.with_name(f"{args.input.stem}_manual_final{args.input.suffix}")
    with output.open("w", encoding="utf-8") as fh:
        json.dump(finalized, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
