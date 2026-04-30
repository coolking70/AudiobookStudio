from __future__ import annotations

import re
from typing import Any

from .alias_registry import AliasRegistry
from .candidate_gen import generate_candidates
from .cleaner import normalize_text
from .consistency_fixer import fix_consistency
from .implicit_attributor import attribute_implicit
from .nlp_backend import HanLPNERBackend, NERBackend
from .quote_extractor import extract_quotes
from .review_router import collect_review_items
from .rule_attributor import attribute_explicit
from .schema import Attribution, AttributionType, CandidateSet, ParseResult, QuoteSpan, SegmentEx
from .spc_ranker import CandidateRanker, OpenAICompatibleSPCRanker


NARRATOR_CUE_ONLY_RE = re.compile(
    r"^[，,。；;：:\s]*[一-龥A-Za-z0-9_]{1,24}"
    r"(?:对|向|冲)?[一-龥A-Za-z0-9_]{0,12}"
    r"(?:喃喃道|嘀咕道|低声道|轻声道|笑道|冷笑道|喝道|骂道|叹道|缓缓开口|开口|心想|想道|说道|说|问道|问|答道|答|喊道|喊|叫道|叫|道)"
    r"[：:，,。\s]*$"
)

# P1b：匹配「我」但排除「我们/我方/我国/我家/我辈/我等」
_FIRST_PERSON_RE = re.compile(r"我(?!们|方|国|家|辈|等)")


def _clean_narrator_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.strip(" \t\r\n")
    if not cleaned:
        return ""
    if NARRATOR_CUE_ONLY_RE.fullmatch(cleaned):
        return ""
    parts = re.split(r"([。！？!?])", cleaned)
    rebuilt: list[str] = []
    for idx in range(0, len(parts), 2):
        sentence = (parts[idx] or "").strip()
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        if not sentence:
            continue
        candidate = f"{sentence}{punct}"
        if NARRATOR_CUE_ONLY_RE.fullmatch(candidate):
            continue
        rebuilt.append(candidate)
    if rebuilt:
        cleaned = " ".join(rebuilt).strip()
    elif NARRATOR_CUE_ONLY_RE.search(cleaned):
        return ""
    return cleaned


def _append_narrator_segments(
    segments: list[SegmentEx],
    raw_text: str,
    quote_id_prefix: str,
    context_chars: int = 120,
) -> None:
    cleaned = _clean_narrator_text(raw_text)
    if not cleaned:
        return
    # 保持段落感，避免把整章旁白塞成单个超长 TTS 片段。
    parts = [part.strip() for part in re.split(r"\n\s*\n+", cleaned) if part.strip()] or [cleaned]
    for idx, part in enumerate(parts, start=1):
        segments.append(
            SegmentEx(
                quote_id=f"{quote_id_prefix}_n{idx}",
                speaker="旁白",
                text=part,
                confidence=1.0,
                evidence="引号外叙述文本",
                attribution_type=AttributionType.NARRATOR,
                candidates=["旁白"],
                candidate_sources={"旁白": ["narrator_gap"]},
                scene_characters=[],
                context_before=part[:context_chars],
                context_after=part[-context_chars:],
            )
        )


# ── 辅助：规范化 role_hints 为扁平列表 ───────────────────────────────────────

def _normalize_role_hints(
    role_hints: dict[str, list[str] | str] | list[str] | None,
) -> list[str]:
    """将各种 role_hints 格式归一为角色名字符串列表。"""
    if not role_hints:
        return []
    if isinstance(role_hints, list):
        return [str(r) for r in role_hints if r]
    if isinstance(role_hints, dict):
        return list(role_hints.keys())
    return []


# ── P1a：自我介绍检测 ────────────────────────────────────────────────────────

def _attribute_self_identified(
    quote: QuoteSpan,
    role_hints: list[str],
) -> Attribution | None:
    """
    检测台词文本中是否含自我介绍模式：
      「我是王冢真唯」「我王冢真唯...」「我叫真唯」等

    仅匹配 role_hints 中的已知角色（全名或后 2/3 字简称）。
    返回 confidence=0.95 的显性归因，不对未知名字猜测。
    """
    if not role_hints:
        return None
    text = quote.text or ""
    if "我" not in text:
        return None

    for role in role_hints:
        aliases_to_check: list[tuple[str, float]] = [(role, 0.95)]  # (alias, confidence)
        if len(role) >= 3:
            aliases_to_check.append((role[-2:], 0.90))
        if len(role) >= 4:
            aliases_to_check.append((role[-3:], 0.90))

        for alias, conf in aliases_to_check:
            for connector in ["", "是", "叫", "乃", "就是"]:
                pattern = f"我{connector}{alias}"
                if pattern in text:
                    return Attribution(
                        quote_id=quote.quote_id,
                        speaker=role,
                        confidence=conf,
                        evidence=f"自我介绍：「{pattern[:18]}」",
                        attribution_type=AttributionType.EXPLICIT_BEFORE,
                    )
    return None


# ── P1b：一人称叙述者锚点 ────────────────────────────────────────────────────

def _attribute_narrator_anchored(
    quote: QuoteSpan,
    narrator: str,
) -> Attribution | None:
    """
    台词含「我」时，归因给主视角叙述者（日式轻小说常用主角固定视角）。

    置信度 0.82：中高置信，允许 fix_consistency 层进一步调整，
    但足够高以压过 implicit 层的随机猜测。
    """
    if _FIRST_PERSON_RE.search(quote.text or ""):
        return Attribution(
            quote_id=quote.quote_id,
            speaker=narrator,
            confidence=0.82,
            evidence="一人称代词「我」→叙述者锚点",
            attribution_type=AttributionType.IMPLICIT,
        )
    return None


# ── P2：对话块检测 ────────────────────────────────────────────────────────────

# 两句台词之间旁白字数超过此值，视为新的对话块
_DIALOGUE_GAP_THRESHOLD = 150


def _build_dialogue_blocks(quotes: list[QuoteSpan]) -> dict[str, str]:
    """
    将 quotes 按文档位置分组为「对话块」。
    相邻两句之间间隙（旁白字数）< _DIALOGUE_GAP_THRESHOLD 时归为同一块。

    Returns:
        {quote_id: "对话块#N 第M/K句"}
        单句独立块（K=1）不生成标注（空字符串），避免噪音。
    """
    if not quotes:
        return {}

    # 逐句分组
    blocks: list[list[QuoteSpan]] = []
    current: list[QuoteSpan] = [quotes[0]]

    for i in range(1, len(quotes)):
        prev_end   = quotes[i - 1].raw_end   if quotes[i - 1].raw_end   is not None else quotes[i - 1].end
        curr_start = quotes[i].raw_start if quotes[i].raw_start is not None else quotes[i].start
        gap = curr_start - prev_end
        if gap <= _DIALOGUE_GAP_THRESHOLD:
            current.append(quotes[i])
        else:
            blocks.append(current)
            current = [quotes[i]]
    blocks.append(current)

    result: dict[str, str] = {}
    for block_id, block in enumerate(blocks, start=1):
        size = len(block)
        for pos, quote in enumerate(block, start=1):
            result[quote.quote_id] = "" if size == 1 else f"对话块#{block_id} 第{pos}/{size}句"
    return result


# ── P0：批量 LLM 归因路径 ────────────────────────────────────────────────────

def _parse_with_batch_llm(
    cleaned: str,
    quotes: list[QuoteSpan],
    aliases: AliasRegistry,
    ner_backend: NERBackend | None,
    batch_llm_config: Any,
    narrator: str | None,
    role_hints_list: list[str],
    return_result: bool,
    review_threshold: float,
    include_narration: bool,
    initial_recent_speakers: list[str] | None = None,
) -> list[SegmentEx] | ParseResult:
    """
    使用 BatchLLMAttributor 的完整归因路径。

    阶段：
      1. 生成所有候选集
      2. 预过滤（P1a 自我介绍 → P1b 叙述者锚点 → 保守规则层）
      3. 批量 LLM 归因（仅发送未解决台词）
      4. 合并归因，构建 SegmentEx 列表
      5. fix_consistency
    """
    from .batch_llm_attributor import BatchLLMAttributor, attribute_explicit_conservative

    # ① 生成所有候选集（传入 initial_recent_speakers，支持章节级累积上下文）
    recent_speakers_acc: list[str] = list(initial_recent_speakers or [])
    all_candidates: dict[str, CandidateSet] = {}
    for quote in quotes:
        cset = generate_candidates(quote, aliases=aliases, nlp_backend=ner_backend,
                                   recent_speakers=recent_speakers_acc)
        all_candidates[quote.quote_id] = cset

    # ② 预过滤：纯规则层（高精确率，低召回率）
    pre_resolved: dict[str, Attribution] = {}
    unresolved_quotes: list[QuoteSpan] = []

    for quote in quotes:
        attr: Attribution | None = None

        # P1a：自我介绍（「我是X」「我X」模式）
        attr = _attribute_self_identified(quote, role_hints_list)

        # P1b：叙述者锚点（「我」→ narrator）
        if attr is None and narrator:
            attr = _attribute_narrator_anchored(quote, narrator)

        # 保守规则层（前/后显性，必须 role_hints 中存在的角色）
        if attr is None:
            attr = attribute_explicit_conservative(quote, role_hints_list)

        if attr is not None:
            pre_resolved[quote.quote_id] = attr
        else:
            unresolved_quotes.append(quote)

    # ③ 批量 LLM 归因
    llm_attributions: dict[str, Attribution] = {}
    if unresolved_quotes:
        # P2：基于所有 quotes（含已预解决的）计算对话块，保留位置关系
        block_hints = _build_dialogue_blocks(quotes)

        attributor = BatchLLMAttributor(batch_llm_config)
        llm_attributions = attributor.attribute(
            unresolved_quotes,
            all_candidates,
            role_hints=role_hints_list,
            block_hints=block_hints,
        )

    # ④ 合并并按文档顺序构建 SegmentEx
    segments: list[SegmentEx] = []
    cursor = 0

    for quote in quotes:
        raw_start = quote.raw_start if quote.raw_start is not None else quote.start
        raw_end   = quote.raw_end   if quote.raw_end   is not None else quote.end

        if include_narration and raw_start > cursor:
            _append_narrator_segments(segments, cleaned[cursor:raw_start], quote.quote_id)

        cset = all_candidates.get(quote.quote_id)

        if quote.quote_id in pre_resolved:
            attribution = pre_resolved[quote.quote_id]
        elif quote.quote_id in llm_attributions:
            attribution = llm_attributions[quote.quote_id]
        else:
            attribution = Attribution(
                quote_id=quote.quote_id,
                speaker="未知",
                confidence=0.20,
                evidence="归因失败（规则+LLM均无结果）",
                attribution_type=AttributionType.UNKNOWN,
            )

        segment = SegmentEx(
            quote_id=quote.quote_id,
            speaker=attribution.speaker,
            text=quote.text,
            addressee=getattr(attribution, "addressee", None) or None,
            confidence=attribution.confidence,
            evidence=attribution.evidence,
            attribution_type=attribution.attribution_type,
            candidates=cset.candidates if cset else [],
            candidate_sources=cset.candidate_sources if cset else {},
            scene_characters=cset.scene_characters if cset else [],
            context_before=quote.context_before,
            context_after=quote.context_after,
        )
        segments.append(segment)
        cursor = max(cursor, raw_end)

    if include_narration and cursor < len(cleaned):
        _append_narrator_segments(segments, cleaned[cursor:], "tail")

    segments = fix_consistency(segments)
    result = ParseResult(
        segments=segments,
        review_items=collect_review_items(segments, threshold=review_threshold),
        stats={
            "quote_count": len(quotes),
            "review_count": sum(1 for s in segments if s.confidence < review_threshold),
            "pre_resolved": len(pre_resolved),
            "llm_resolved": len(llm_attributions),
            "batch_llm_enabled": True,
        },
    )
    return result if return_result else result.segments


# ── 主入口 ────────────────────────────────────────────────────────────────────

def parse_novel(
    text: str,
    role_hints: dict[str, list[str] | str] | list[str] | None = None,
    llm_config: Any | None = None,
    batch_llm_config: Any | None = None,
    narrator: str | None = None,
    return_result: bool = False,
    review_threshold: float = 0.7,
    use_hanlp: bool = False,
    hanlp_model: str | None = None,
    nlp_backend: NERBackend | None = None,
    enable_alias_inference: bool = True,
    implicit_strategy: str = "heuristic",
    implicit_ranker: CandidateRanker | None = None,
    include_narration: bool = False,
    initial_recent_speakers: list[str] | None = None,
) -> list[SegmentEx] | ParseResult:
    """Parse Chinese novel text into speaker-attributed TTS segments.

    Args:
        text:               原始小说文本。
        role_hints:         已知角色名（列表或 {名: [别名]} 字典）。
        llm_config:         （兼容旧接口）逐条 LLM 配置，不再推荐使用。
        batch_llm_config:   BatchConfig 实例，启用后走批量 LLM 归因路径（推荐）。
        narrator:           主视角叙述者角色名，设置后台词中的「我」自动锚点到该角色。
                            需配合 role_hints 包含该角色名使用。
        return_result:      为 True 时返回 ParseResult，否则返回 list[SegmentEx]。
        review_threshold:   低于该置信度的片段被收集到 review_items。
        include_narration:  是否在 segments 中插入旁白片段。
        （其余参数控制 NER 后端和隐式归因策略，仅在非 batch_llm 路径下生效。）
    """
    cleaned = normalize_text(text)
    aliases = AliasRegistry.from_role_hints(role_hints)
    ner_backend = nlp_backend or (HanLPNERBackend(hanlp_model) if use_hanlp else None)
    quotes = extract_quotes(cleaned)
    role_hints_list = _normalize_role_hints(role_hints)

    # ── 别名推断（通用，两条路径共用）────────────────────────────────────────
    inferred_aliases: dict[str, str] = {}
    if enable_alias_inference and quotes:
        observed_names: list[str] = []
        for quote in quotes:
            candidate_set = generate_candidates(quote, aliases=aliases, nlp_backend=ner_backend)
            observed_names.extend(candidate_set.candidates)
        inferred_aliases = aliases.update_inferred_aliases(observed_names)

    # ── 批量 LLM 路径（P0）────────────────────────────────────────────────────
    if batch_llm_config is not None:
        return _parse_with_batch_llm(
            cleaned=cleaned,
            quotes=quotes,
            aliases=aliases,
            ner_backend=ner_backend,
            batch_llm_config=batch_llm_config,
            narrator=narrator,
            role_hints_list=role_hints_list,
            return_result=return_result,
            review_threshold=review_threshold,
            include_narration=include_narration,
            initial_recent_speakers=initial_recent_speakers,
        )

    # ── 经典逐条路径（兼容旧接口）──────────────────────────────────────────────
    segments: list[SegmentEx] = []
    recent_speakers: list[str] = list(initial_recent_speakers or [])
    ranker = implicit_ranker
    if ranker is None and implicit_strategy == "llm_spc":
        if llm_config is None:
            raise RuntimeError("llm_config is required when implicit_strategy='llm_spc'")
        ranker = OpenAICompatibleSPCRanker(llm_config)

    cursor = 0
    for quote in quotes:
        raw_start = quote.raw_start if quote.raw_start is not None else quote.start
        raw_end   = quote.raw_end   if quote.raw_end   is not None else quote.end
        if include_narration and raw_start > cursor:
            _append_narrator_segments(segments, cleaned[cursor:raw_start], quote.quote_id)

        candidate_set = generate_candidates(
            quote,
            aliases=aliases,
            recent_speakers=recent_speakers,
            nlp_backend=ner_backend,
        )

        # P1a：自我介绍优先
        attribution = _attribute_self_identified(quote, role_hints_list)

        # P1b：叙述者锚点
        if attribution is None and narrator:
            attribution = _attribute_narrator_anchored(quote, narrator)

        # 规则显性归因
        if attribution is None:
            attribution = attribute_explicit(quote, aliases=aliases)

        if attribution is None:
            attribution = attribute_implicit(
                quote, candidate_set,
                recent_speakers=recent_speakers,
                ranker=ranker,
            )
        elif attribution.speaker in {"他", "她"}:
            if ranker is not None:
                attribution = ranker.rank(quote, candidate_set, recent_speakers=recent_speakers)
                attribution.attribution_type = AttributionType.LATENT
            elif recent_speakers:
                attribution.speaker = recent_speakers[-1]
                attribution.confidence = min(attribution.confidence, 0.72)
                attribution.evidence = f"{attribution.evidence}；代词承接最近说话人"

        segment = SegmentEx(
            quote_id=quote.quote_id,
            speaker=attribution.speaker,
            text=quote.text,
            addressee=getattr(attribution, "addressee", None) or None,
            confidence=attribution.confidence,
            evidence=attribution.evidence,
            attribution_type=attribution.attribution_type,
            candidates=candidate_set.candidates,
            candidate_sources=candidate_set.candidate_sources,
            scene_characters=candidate_set.scene_characters,
            context_before=quote.context_before,
            context_after=quote.context_after,
        )
        segments.append(segment)
        if segment.speaker not in {"旁白", "未知"}:
            recent_speakers.append(segment.speaker)
        cursor = max(cursor, raw_end)

    if include_narration and cursor < len(cleaned):
        _append_narrator_segments(segments, cleaned[cursor:], "tail")

    segments = fix_consistency(segments)
    result = ParseResult(
        segments=segments,
        review_items=collect_review_items(segments, threshold=review_threshold),
        stats={
            "quote_count": len(quotes),
            "review_count": sum(1 for item in segments if item.confidence < review_threshold),
            "hanlp_enabled": ner_backend is not None,
            "inferred_aliases": inferred_aliases,
            "implicit_strategy": implicit_strategy,
        },
    )
    return result if return_result else result.segments
