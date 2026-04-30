from __future__ import annotations

import re

from .schema import Attribution, AttributionType, CandidateSet, QuoteSpan
from .spc_ranker import CandidateRanker


OBSERVE_VERBS = ("看见", "望见", "看向", "望向")
LOW_VALUE = {"旁白", "未知", "众人", "大家", "二人", "三人", "所有人", "公子", "姑娘", "师姐", "师兄", "师妹"}


def _usable_candidates(candidates: CandidateSet) -> list[str]:
    return [name for name in candidates.candidates if name not in LOW_VALUE]


def _last_sentence(text: str) -> str:
    parts = re.split(r"[。！？!?\n]", text.strip())
    return next((part.strip() for part in reversed(parts) if part.strip()), "")


def _observer_before_quote(quote: QuoteSpan, usable: list[str]) -> str | None:
    sentence = _last_sentence(quote.context_before)
    if not sentence:
        return None
    verb_positions = [sentence.find(verb) for verb in OBSERVE_VERBS if verb in sentence]
    if not verb_positions:
        return None
    cue_pos = min(verb_positions)
    before_cue = sentence[:cue_pos]
    for name in sorted(usable, key=len, reverse=True):
        if name in before_cue:
            return name
    return None


def _observed_target_in_context(quote: QuoteSpan, usable: list[str]) -> str | None:
    context = quote.context_before[-160:]
    best: tuple[int, str] | None = None
    for verb in OBSERVE_VERBS:
        start = 0
        while True:
            pos = context.find(verb, start)
            if pos == -1:
                break
            after_cue = context[pos + len(verb):pos + len(verb) + 24]
            for name in sorted(usable, key=len, reverse=True):
                name_pos = after_cue.find(name)
                if name_pos != -1:
                    absolute_pos = pos + len(verb) + name_pos
                    if best is None or absolute_pos > best[0]:
                        best = (absolute_pos, name)
            start = pos + len(verb)
    return best[1] if best else None


def attribute_implicit(
    quote: QuoteSpan,
    candidates: CandidateSet,
    recent_speakers: list[str] | None = None,
    ranker: CandidateRanker | None = None,
) -> Attribution:
    """Heuristic placeholder for future SPC/CSI model integration."""

    recent_speakers = recent_speakers or []
    usable = _usable_candidates(candidates)

    observer = _observer_before_quote(quote, usable)
    if observer:
        return Attribution(
            quote_id=quote.quote_id,
            speaker=observer,
            confidence=0.56,
            evidence="观察者动作启发式",
            attribution_type=AttributionType.IMPLICIT,
        )

    if len(usable) == 1:
        return Attribution(
            quote_id=quote.quote_id,
            speaker=usable[0],
            confidence=0.62,
            evidence="候选集中只有一个非旁白角色",
            attribution_type=AttributionType.IMPLICIT,
        )

    if ranker is not None:
        return ranker.rank(quote, candidates, recent_speakers=recent_speakers)

    if not recent_speakers and usable:
        first = usable[0]
        return Attribution(
            quote_id=quote.quote_id,
            speaker=first,
            confidence=0.42,
            evidence="首轮候选排序启发式",
            attribution_type=AttributionType.IMPLICIT,
        )

    if recent_speakers and usable:
        last = recent_speakers[-1]
        observed_target = _observed_target_in_context(quote, usable)
        if observed_target and observed_target != last:
            return Attribution(
                quote_id=quote.quote_id,
                speaker=observed_target,
                confidence=0.6,
                evidence="观察目标轮替启发式",
                attribution_type=AttributionType.IMPLICIT,
            )
        alternate = next((name for name in usable if name != last), None)
        if alternate:
            return Attribution(
                quote_id=quote.quote_id,
                speaker=alternate,
                confidence=0.58,
                evidence="连续双人对话轮替启发式",
                attribution_type=AttributionType.IMPLICIT,
            )

    return Attribution(
        quote_id=quote.quote_id,
        speaker="未知",
        confidence=0.2,
        evidence="未发现显式说话人，等待复核",
        attribution_type=AttributionType.UNKNOWN,
    )
