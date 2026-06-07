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

from .batch_llm_attributor import BatchConfig, BatchLLMAttributor
from .schema import AttributionType, CandidateSet as BatchCandidateSet, QuoteSpan as BatchQuoteSpan, SegmentEx
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
    "address_term_backcheck",
    "address_term_local_context",
    "scene_active",
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
GENERIC_REVIEW_SPEAKERS = {
    "少女",
    "女孩子",
    "女性",
    "旁边的孩子",
    "未命名人物",
    "其他角色",
    "三位女性",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "妈妈",
    "母亲",
    "爸爸",
    "父亲",
}

BATCH_REVIEW_EVIDENCE_MARKERS = (
    "LLM复核待人工",
    "称呼反推",
    "称呼负权重",
    "泛称",
    "关系称谓",
    "待人工",
)
FIRST_PERSON_ANCHOR_MARKERS = (
    "叙述者",
    "内心独白",
    "自白",
    "前文「我",
    "前文“我",
    "后文明确“我",
    "紧前文“我",
    "“我”为",
    "「我”为",
    "前文为叙述者",
    "后文“我”",
)

SPEECH_EVIDENCE_MARKERS = (
    "说道",
    "说",
    "道",
    "问",
    "回答",
    "答",
    "开口",
    "喊",
    "叫",
    "前文「",
    "后文「",
    "紧前文「",
    "称呼反推",
    "动作主语",
)


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
    local_model_path: str | None = None
    local_runtime: str | None = None
    local_engine: str | None = None
    local_device: str | None = None
    local_ctx_tokens: int | None = None
    local_gpu_layers: int | None = None
    local_threads: int | None = None
    local_batch_size: int | None = None


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


def _has_first_person_anchor(seg: SegmentEx) -> bool:
    evidence = str(seg.evidence or "")
    if any(marker in evidence for marker in FIRST_PERSON_ANCHOR_MARKERS):
        return True
    context = f"{seg.context_before[-120:]}\n{seg.context_after[:120]}"
    return bool(
        ("我" in context)
        and any(marker in evidence for marker in ("前文", "后文", "紧前文", "内心", "自白", "叙述者"))
    )


def _is_turn_only_llm_evidence(evidence: str) -> bool:
    evidence = str(evidence or "")
    if not any(marker in evidence for marker in ("轮换", "上一句", "上句", "对话轮")):
        return False
    return not any(marker in evidence for marker in SPEECH_EVIDENCE_MARKERS)


def _segment_to_batch_quote_span(seg: SegmentEx) -> BatchQuoteSpan:
    return BatchQuoteSpan(
        quote_id=seg.quote_id or "",
        text=seg.text,
        start=0,
        end=len(seg.text),
        context_before=seg.context_before,
        context_after=seg.context_after,
        raw=seg.text,
    )


def _segment_to_batch_candidate_set(
    seg: SegmentEx,
    role_hints: list[str],
    narrator: str | None = None,
) -> BatchCandidateSet:
    candidates: list[str] = []
    for name in list(seg.candidates or []) + list(seg.scene_characters or []) + list(role_hints or []) + [narrator]:
        value = str(name or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    candidate_sources = {key: list(value) for key, value in (seg.candidate_sources or {}).items()}
    if narrator:
        candidate_sources.setdefault(narrator, [])
        if "narrator_anchor" not in candidate_sources[narrator]:
            candidate_sources[narrator].append("narrator_anchor")
    for fallback in ("旁白", "未知"):
        if fallback not in candidates:
            candidates.append(fallback)
    return BatchCandidateSet(
        quote_id=seg.quote_id or "",
        candidates=candidates,
        candidate_sources=candidate_sources,
        scene_characters=seg.scene_characters or [],
    )


def _looks_like_review_target(seg: SegmentEx, threshold: float) -> bool:
    speaker = str(seg.speaker or "").strip()
    evidence = str(seg.evidence or "")
    try:
        confidence = float(seg.confidence if seg.confidence is not None else 1.0)
    except Exception:
        confidence = 1.0
    return (
        confidence < threshold
        or speaker in SKIP_SPEAKERS
        or speaker in GENERIC_REVIEW_SPEAKERS
        or any(marker in evidence for marker in BATCH_REVIEW_EVIDENCE_MARKERS)
    )


def _derive_role_hints(segments: list[SegmentEx]) -> list[str]:
    names: list[str] = []
    for seg in segments:
        for name in [seg.speaker] + list(seg.scene_characters or []) + list(seg.candidates or []):
            value = str(name or "").strip()
            if not value or value in SKIP_SPEAKERS or value in GENERIC_REVIEW_SPEAKERS:
                continue
            if value not in names:
                names.append(value)
    return names


def _recent_confirmed_speakers_before(
    segments: list[SegmentEx],
    index: int,
    window: int = 4,
) -> list[str]:
    recent: list[str] = []
    for seg in segments[max(0, index - 24):index]:
        speaker = str(seg.speaker or "").strip()
        if speaker and speaker not in SKIP_SPEAKERS:
            recent.append(speaker)
    return recent[-window:]


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
        evidence = str(original.evidence or "")
        sources = _candidate_sources(original, new_speaker)
        if sources <= {"narrator_anchor"} and original.confidence < DEFAULT_THRESHOLD:
            return False, "原结果低置信且复核仅有叙述者锚点，需人工确认"
        if new_speaker in GENERIC_REVIEW_SPEAKERS:
            if new_confidence >= 0.90:
                return True, ""
            return False, "当前说话人是泛称/关系称谓，需人工确认是否映射到具体角色"
        if ("称呼反推" in evidence and "冲突" in evidence) or "称呼负权重" in evidence:
            return False, "原结果存在称呼负权重/反推冲突，需人工确认"
        if ("场景约束" in evidence or "跳变" in evidence) and not _has_strong_candidate_support(original, new_speaker):
            return False, "原结果存在场景约束/跳变风险，需人工确认"
        return True, ""
    sources = _candidate_sources(original, new_speaker)
    evidence_text = str(original.evidence or "")
    address_conflict = ("称呼反推" in evidence_text and "冲突" in evidence_text) or "称呼负权重" in evidence_text
    if address_conflict and "address_term_backcheck" in sources and new_confidence >= 0.88:
        return True, ""
    if original.speaker not in SKIP_SPEAKERS and _has_first_person_anchor(original):
        return False, "原结果存在第一人称/叙述者锚点，需人工确认后再改写"
    if new_confidence < MIN_AUTO_CORRECT_CONFIDENCE:
        return False, f"复核置信度 {new_confidence:.2f} 低于自动覆盖阈值"
    if not _has_strong_candidate_support(original, new_speaker):
        sources = sorted(_candidate_sources(original, new_speaker))
        source_text = "、".join(sources) if sources else "无候选来源"
        return False, f"复核目标缺少场景强支持（{source_text}）"
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

        if new_speaker and new_speaker == seg.speaker and can_apply:
            seg.confidence = max(seg.confidence, min(max(new_confidence, threshold + 0.15), 0.95))
            seg.evidence = f"{seg.evidence or ''}；LLM确认({new_confidence:.2f})"
            stats["confirmed"] += 1
        elif new_speaker and new_speaker != seg.speaker and can_apply:
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
        elif new_speaker and not can_apply:
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


def route_to_batch_llm(
    segments: list[SegmentEx],
    llm_config: LLMRouterConfig | Any,
    threshold: float = DEFAULT_THRESHOLD,
    review_indices: list[int] | None = None,
    batch_size: int = 8,
    narrator: str | None = None,
    verbose: bool = False,
) -> tuple[list[SegmentEx], dict[str, Any]]:
    """Batch-review unresolved speaker attributions with dialogue context.

    This is intentionally stricter than the main BatchLLM pass: it only
    rewrites a segment when the batch model's answer passes the same local
    evidence gate used by route_to_llm(). Otherwise the model suggestion is
    preserved in evidence for human review.
    """
    updated = [seg.model_copy() for seg in segments]
    role_hints = _derive_role_hints(updated)
    narrator = str(narrator or "").strip() or None
    if narrator and narrator not in role_hints:
        role_hints.insert(0, narrator)
    review_index_set = set(review_indices) if review_indices is not None else None
    target_indices = [
        idx for idx, seg in enumerate(updated)
        if 0 <= idx < len(updated)
        and (review_index_set is None or idx in review_index_set)
        and _looks_like_review_target(seg, threshold)
    ]
    stats: dict[str, Any] = {
        "mode": "batch",
        "threshold": threshold,
        "targets": len(target_indices),
        "reviewed": 0,
        "corrected": 0,
        "confirmed": 0,
        "failed": 0,
        "blocked": 0,
        "skipped": len(updated) - len(target_indices),
        "role_hints_count": len(role_hints),
        "batch_size": batch_size,
    }
    if not target_indices:
        return updated, stats

    batch_config = BatchConfig(
        base_url=str(getattr(llm_config, "base_url", "") or "http://127.0.0.1:1234/v1"),
        api_key=str(getattr(llm_config, "api_key", "") or "local"),
        model=str(getattr(llm_config, "model", "") or "local-model"),
        batch_size=max(1, min(int(batch_size or 8), 12)),
        max_tokens=max(int(getattr(llm_config, "max_tokens", 4096) or 4096), 1024),
        temperature=float(getattr(llm_config, "temperature", 0.0) or 0.0),
        timeout=int(getattr(llm_config, "timeout", 180) or 180),
        context_chars=260,
        output_mode="compact",
        max_retries=1,
        disable_thinking=True,
    )
    attributor = BatchLLMAttributor(batch_config)

    for batch_start in range(0, len(target_indices), batch_config.batch_size):
        batch_indices = target_indices[batch_start: batch_start + batch_config.batch_size]
        batch: list[tuple[BatchQuoteSpan, BatchCandidateSet]] = []
        for idx in batch_indices:
            seg = updated[idx]
            batch.append((
                _segment_to_batch_quote_span(seg),
                _segment_to_batch_candidate_set(seg, role_hints, narrator=narrator),
            ))

        prev_speakers = _recent_confirmed_speakers_before(updated, batch_indices[0])
        try:
            attributions = attributor.attribute_batch(
                batch,
                role_hints=role_hints,
                prev_speakers=prev_speakers,
                narrator=narrator,
            )
        except Exception as exc:
            stats["failed"] += len(batch_indices)
            logger.warning("[review_router] BatchLLM review failed for %s items: %s", len(batch_indices), exc)
            if verbose:
                print(f"[review_router] BatchLLM review failed for {len(batch_indices)} items: {exc}")
            continue

        attr_map = {attr.quote_id: attr for attr in attributions}
        for idx in batch_indices:
            seg = updated[idx]
            attr = attr_map.get(seg.quote_id or "")
            stats["reviewed"] += 1
            if not attr:
                stats["failed"] += 1
                seg.evidence = f"{seg.evidence or ''}；BatchLLM复核失败: 未返回 {seg.quote_id}"
                continue

            new_speaker = str(attr.speaker or "").strip()
            if narrator and (
                new_speaker in {"叙述者", "我", "第一人称叙述者", "主叙述者"}
                or "叙述者" in new_speaker
            ):
                new_speaker = narrator
            new_confidence = float(attr.confidence or 0.0)
            if narrator and new_speaker == narrator:
                seg.candidate_sources = {key: list(value) for key, value in (seg.candidate_sources or {}).items()}
                seg.candidate_sources.setdefault(narrator, [])
                if "narrator_anchor" not in seg.candidate_sources[narrator]:
                    seg.candidate_sources[narrator].append("narrator_anchor")
            turn_only_evidence = _is_turn_only_llm_evidence(str(attr.evidence or ""))
            can_apply, block_reason = _should_auto_apply_review(seg, new_speaker, new_confidence)
            if can_apply and turn_only_evidence and seg.confidence < threshold:
                can_apply = False
                block_reason = "复核依据仅为上一句/轮换，低置信片段需人工确认"

            if new_speaker == seg.speaker and can_apply:
                seg.confidence = max(seg.confidence, min(max(new_confidence, threshold + 0.12), 0.95))
                seg.evidence = f"{seg.evidence or ''}；BatchLLM确认({new_confidence:.2f}): {attr.evidence or ''}"
                stats["confirmed"] += 1
            elif new_speaker and new_speaker != seg.speaker and can_apply:
                if verbose:
                    print(
                        f"[review_router] BatchLLM {seg.quote_id} "
                        f"{seg.speaker!r}({seg.confidence:.2f}) -> "
                        f"{new_speaker!r}({new_confidence:.2f})"
                    )
                seg.speaker = new_speaker
                seg.confidence = new_confidence
                seg.attribution_type = AttributionType.IMPLICIT
                seg.evidence = f"BatchLLM复核: {attr.evidence or ''}"
                stats["corrected"] += 1
            elif new_speaker:
                seg.evidence = (
                    f"{seg.evidence or ''}；BatchLLM复核待人工: "
                    f"建议 {new_speaker}({new_confidence:.2f})，{block_reason}，依据: {attr.evidence or ''}"
                )
                seg.confidence = min(seg.confidence, threshold - 0.01)
                stats["blocked"] += 1
            else:
                seg.evidence = f"{seg.evidence or ''}；BatchLLM复核失败: 未给出明确说话人"
                stats["failed"] += 1

    return updated, stats
