from __future__ import annotations

import re

from .schema import QuoteSpan


BRACKET_QUOTE_RE = re.compile(r"「(?P<bracket>[^「」]{1,1000})」|\"(?P<ascii>[^\"\n]{1,300})\"")
COLON_QUOTE_RE = re.compile(
    r"(?P<cue>[\u4e00-\u9fa5A-Za-z0-9_]{1,24}?(?:说|说道|问|问道|答|答道|喊|喊道|道|心想|想道)[：:])"
    r"(?P<quote>[^。！？!?\n「」\"]{1,240}[。！？!?])"
)


def _window(text: str, start: int, end: int, size: int) -> tuple[str, str]:
    before = text[max(0, start - size):start].strip()
    after = text[end:min(len(text), end + size)].strip()
    return before, after


def extract_quotes(text: str, context_chars: int = 120, prefix: str = "q") -> list[QuoteSpan]:
    extracted: list[tuple[int, int, int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    for match in BRACKET_QUOTE_RE.finditer(text):
        quote_text = (match.group("bracket") or match.group("ascii") or "").strip()
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
