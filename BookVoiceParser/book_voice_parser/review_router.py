"""
review_router.py
----------------
两个职责：
  1. collect_review_items()  —— 收集低置信度片段（原有逻辑，不变）
  2. route_to_llm()          —— 把低置信度片段批量送给本地 LLM 做 SPC 复核，
                                回写结果到 SegmentEx 列表。
                                复用 spc_ranker.py 的 prompt 基础设施。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from .schema import AttributionType, SegmentEx
from .spc_ranker import (
    CandidateSet,
    OpenAICompatibleSPCRanker,
    QuoteSpan,
    build_spc_task,
    build_spc_prompt,
)


# ── 默认复核阈值 ───────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.7
SKIP_SPEAKERS = {"", "旁白", "未知", "UNKNOWN"}
MIN_AUTO_CORRECT_CONFIDENCE = 0.75
MIN_AUTO_CORRECT_MARGIN = 0.08

# These sources mean the speaker was supported by local context, explicit
# relation-role logic, or the active narrator anchor. Fallback-only candidates
# are useful for LLM choice, but too noisy to auto-overwrite existing labels.
STRONG_CANDIDATE_SOURCES = {
    "role_hints",
    "appearance_alias",
    "group_cue",
    "rule_cue",
    "title",
    "hanlp_ner",
    "relation_conditional",
}
MODERATE_CANDIDATE_SOURCES = {
    "relation_inferred",
    "narrator_anchor",
}
WEAK_CANDIDATE_SOURCES = {
    "role_hints_fallback",
    "recent_speakers_extended",
    "recent_speakers",
    "relation_mention",
}


# ── LLM 连接配置（dict 或 dataclass 均可，与 spc_ranker 兼容同一接口）──────
@dataclass
class LLMRouterConfig:
    """
    轻量配置对象，字段名与 spc_ranker 的 _config_value() 兼容。
    也可以直接传 AudiobookStudio 的 LLMConfig 对象（有 base_url/model/api_key）。
    """
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen/qwen3.6-35b-a3b"
    api_key: str = "lm-studio"
    # Qwen3 thinking 模型约需 700 reasoning tokens，1024 是安全下限
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: int = 60


def _segment_to_quote_span(seg: SegmentEx) -> QuoteSpan:
    """把 SegmentEx 重新包装成 QuoteSpan，用于 build_spc_task()。"""
    return QuoteSpan(
        quote_id=seg.quote_id or "",
        text=seg.text,
        start=0,
        end=len(seg.text),
        context_before=seg.context_before,
        context_after=seg.context_after,
    )


def _segment_to_candidate_set(seg: SegmentEx) -> CandidateSet:
    """把 SegmentEx 里的候选信息包装成 CandidateSet。"""
    return CandidateSet(
        quote_id=seg.quote_id or "",
        candidates=seg.candidates or ["旁白", "未知"],
        candidate_sources=seg.candidate_sources or {},
        scene_characters=seg.scene_characters or [],
    )


def _candidate_sources(seg: SegmentEx, speaker: str) -> set[str]:
    sources = seg.candidate_sources or {}
    return {str(item) for item in sources.get(speaker, []) if item}


def _has_strong_candidate_support(seg: SegmentEx, speaker: str) -> bool:
    if speaker in SKIP_SPEAKERS:
        return False
    sources = _candidate_sources(seg, speaker)
    if sources & STRONG_CANDIDATE_SOURCES:
        return True
    if sources & MODERATE_CANDIDATE_SOURCES:
        return True
    if sources and sources <= WEAK_CANDIDATE_SOURCES:
        return False
    # No source metadata usually means an old snapshot. In that case, treat
    # scene_characters as moderate support, but still require high confidence.
    return speaker in set(seg.scene_characters or [])


def _should_auto_apply_review(
    original: SegmentEx,
    new_speaker: str,
    new_confidence: float,
) -> tuple[bool, str]:
    """Return whether an LLM review correction is safe enough to overwrite.

    The review model is allowed to propose corrections from broad fallback
    candidates, but automatic writes require local scene evidence. Otherwise
    the segment stays unchanged and is marked for human/local-model review.
    """
    if not new_speaker or new_speaker in SKIP_SPEAKERS:
        return False, "复核未给出明确说话人"
    if new_speaker == original.speaker:
        return True, ""
    if new_confidence < MIN_AUTO_CORRECT_CONFIDENCE:
        return False, f"复核置信度 {new_confidence:.2f} 低于自动覆盖阈值"
    if not _has_strong_candidate_support(original, new_speaker):
        sources = sorted(_candidate_sources(original, new_speaker))
        source_text = "、".join(sources) if sources else "无候选来源"
        return False, f"复核目标缺少场景强支持（{source_text}）"
    sources = _candidate_sources(original, new_speaker)
    if new_speaker != original.speaker and sources <= {"narrator_anchor"}:
        return False, "复核目标仅有叙述者锚点支持"
    if (
        new_speaker != original.speaker
        and sources <= {"relation_inferred"}
        and original.speaker not in SKIP_SPEAKERS
    ):
        return False, "复核目标仅有关系称谓推断支持"
    if original.speaker not in SKIP_SPEAKERS and original.confidence >= 0.85:
        if sources & STRONG_CANDIDATE_SOURCES and new_confidence >= 0.88:
            return True, ""
        margin = new_confidence - original.confidence
        if margin < MIN_AUTO_CORRECT_MARGIN:
            return False, f"原结果非低置信，复核优势不足（+{margin:.2f}）"
    return True, ""


# ── 原有接口（不变） ──────────────────────────────────────────────────────
def collect_review_items(
    segments: list[SegmentEx],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """收集置信度低于 threshold 的片段，供人工或 LLM 复核。"""
    items: list[dict[str, Any]] = []
    for segment in segments:
        if segment.confidence >= threshold:
            continue
        items.append(
            {
                "quote_id": segment.quote_id,
                "text": segment.text,
                "speaker": segment.speaker,
                "confidence": segment.confidence,
                "candidates": segment.candidates,
                "candidate_sources": segment.candidate_sources,
                "scene_characters": segment.scene_characters,
                "evidence": segment.evidence,
            }
        )
    return items


# ── 新增：LLM 批量复核并回写 ──────────────────────────────────────────────
def route_to_llm(
    segments: list[SegmentEx],
    llm_config: LLMRouterConfig | Any,
    threshold: float = DEFAULT_THRESHOLD,
    recent_speakers_window: int = 4,
    verbose: bool = False,
) -> tuple[list[SegmentEx], dict[str, Any]]:
    """
    对 confidence < threshold 的片段调用本地 LLM（SPC 风格）进行复核，
    把结果直接回写进 segments 的副本。

    参数
    ----
    segments        : parse_novel() 返回的 SegmentEx 列表
    llm_config      : LLMRouterConfig 或任何有 base_url/model/api_key 属性的对象
    threshold       : 低于此置信度才触发 LLM 复核（默认 0.7）
    recent_speakers_window : 复核时回溯多少个已确定说话人作为上下文
    verbose         : 打印每次复核的结果

    返回
    ----
    (updated_segments, stats)
    stats 包含：total_reviewed / corrected / confirmed / failed / skipped
    """
    ranker = OpenAICompatibleSPCRanker(llm_config)
    updated = [seg.model_copy() for seg in segments]

    stats: dict[str, Any] = {
        "total_reviewed": 0,
        "corrected": 0,
        "confirmed": 0,
        "failed": 0,
        "skipped": 0,
        "blocked": 0,
        "threshold": threshold,
    }

    # 维护一个滑动窗口的已确定说话人列表（用于 recent_speakers prompt）
    confirmed_speakers: list[str] = []

    for idx, seg in enumerate(updated):
        if seg.confidence >= threshold:
            stats["skipped"] += 1
            if seg.speaker not in {"旁白", "未知"}:
                confirmed_speakers.append(seg.speaker)
            continue

        stats["total_reviewed"] += 1
        recent = confirmed_speakers[-recent_speakers_window:]

        quote_span = _segment_to_quote_span(seg)
        candidate_set = _segment_to_candidate_set(seg)
        task = build_spc_task(quote_span, candidate_set, recent_speakers=recent)
        messages = build_spc_prompt(task)

        try:
            attribution = ranker.rank(quote_span, candidate_set, recent_speakers=recent)
        except Exception as exc:
            logger.warning(f"[review_router] LLM call failed for {seg.quote_id}: {exc}")
            if verbose:
                print(f"[review_router] LLM call failed for {seg.quote_id}: {exc}")
            stats["failed"] += 1
            continue

        new_speaker = attribution.speaker
        new_confidence = attribution.confidence

        can_apply, block_reason = _should_auto_apply_review(seg, new_speaker, new_confidence)

        if new_speaker and new_speaker != seg.speaker and can_apply:
            if verbose:
                print(
                    f"[review_router] {seg.quote_id} "
                    f"{seg.speaker!r}({seg.confidence:.2f}) → "
                    f"{new_speaker!r}({new_confidence:.2f})  "
                    f"ev: {attribution.evidence}"
                )
            seg.speaker = new_speaker
            seg.confidence = new_confidence
            seg.attribution_type = AttributionType.IMPLICIT
            seg.evidence = f"LLM复核: {attribution.evidence or ''}"
            stats["corrected"] += 1
        elif new_speaker and new_speaker != seg.speaker:
            if verbose:
                print(
                    f"[review_router] {seg.quote_id} blocked "
                    f"{seg.speaker!r}({seg.confidence:.2f}) → "
                    f"{new_speaker!r}({new_confidence:.2f}): {block_reason}"
                )
            seg.evidence = (
                f"{seg.evidence or ''}；LLM复核待人工: "
                f"建议 {new_speaker}({new_confidence:.2f})，{block_reason}"
            )
            seg.confidence = min(seg.confidence, threshold - 0.01)
            stats["blocked"] += 1
        else:
            # LLM 确认原归属，小幅提升置信度
            seg.confidence = min(seg.confidence + 0.12, 0.85)
            seg.evidence = f"{seg.evidence or ''}；LLM确认({new_confidence:.2f})"
            stats["confirmed"] += 1

        if seg.speaker not in {"旁白", "未知"}:
            confirmed_speakers.append(seg.speaker)

    return updated, stats
