from __future__ import annotations

import re

from .schema import QuoteSpan


BRACKET_QUOTE_RE = re.compile(r"「(?P<bracket>[^「」]{1,1000})」|\"(?P<ascii>[^\"\n]{1,300})\"")
COLON_QUOTE_RE = re.compile(
    r"(?P<cue>[一-龥A-Za-z0-9_]{1,24}?(?:说|说道|问|问道|答|答道|喊|喊道|道|心想|想道)[：:])"
    r"(?P<quote>[^。！？!?\n「」\"]{1,240}[。！？!?])"
)

# 」直接后跟以下内容时，该引号是名词性/描述性引用，不是台词：
#   的（排除 的话/的是/的都/的有/的吗 等口语助词，这些后面可能还是对话）
#   这个 / 那个
#   了?一声（如「唉」了一声 / 「啊」一声）
_NOMINAL_AFTER_RE = re.compile(
    r"^(?:"
    r"的(?![话是都有吗呢啊嘛了吧哦哟])"
    r"|这个|那个"
    r"|了?一声"
    r")"
)


def _window(text: str, start: int, end: int, size: int) -> tuple[str, str]:
    before = text[max(0, start - size):start].strip()
    after = text[end:min(len(text), end + size)].strip()
    return before, after


def _is_nominal_quote(raw_end: int, full_text: str) -> bool:
    """返回 True 表示该引号是名词/描述性用法，不应提取为台词。"""
    after = full_text[raw_end:raw_end + 4].lstrip()
    return bool(_NOMINAL_AFTER_RE.match(after))


def extract_quotes(text: str, context_chars: int = 120, prefix: str = "q") -> list[QuoteSpan]:
    extracted: list[tuple[int, int, int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    for match in BRACKET_QUOTE_RE.finditer(text):
        quote_text = (match.group("bracket") or match.group("ascii") or "").strip()
        # 过滤名词性引用（「X」的声音 / 「X」这个 / 「X」一声）
        if _is_nominal_quote(match.end(), text):
            continue
        extracted.append((match.start(), match.end(), match.start(), match.end(), quote_text, match.group(0)))
        occupied.append((match.start(), match.end()))

    for match in COLON_QUOTE_RE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        quote_text = match.group("quote").strip()
        extracted.append((match.start(), match.end(), match.start("quote"), match.end("quote"), quote_text, match.group(0)))

    quotes: list[QuoteSpan] = []
    for idx, (raw_start, raw_end, start, end, quote_text, raw) in enumerate(sorted(extracted), start=1):
        before, after = _window(text, start, end, context_chars)
        quotes.append(
            QuoteSpan(
                quote_id=f"{prefix}{idx:04d}",
                text=quote_text,
                start=start,
                end=end,
                raw_start=raw_start,
                raw_end=raw_end,
                context_before=before,
                context_after=after,
                raw=raw,
            )
        )
    return quotes
