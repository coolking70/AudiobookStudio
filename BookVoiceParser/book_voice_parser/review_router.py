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

from dataclasses import dataclass
from typing import Any

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
            if verbose:
                print(f"[review_router] LLM call failed for {seg.quote_id}: {exc}")
            stats["failed"] += 1
            continue

        new_speaker = attribution.speaker
        new_confidence = attribution.confidence

        if new_speaker and new_speaker != seg.speaker and new_speaker != "未知":
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
        else:
            # LLM 确认原归属，小幅提升置信度
            seg.confidence = min(seg.confidence + 0.12, 0.85)
            seg.evidence = f"{seg.evidence or ''}；LLM确认({new_confidence:.2f})"
            stats["confirmed"] += 1

        if seg.speaker not in {"旁白", "未知"}:
            confirmed_speakers.append(seg.speaker)

    return updated, stats
