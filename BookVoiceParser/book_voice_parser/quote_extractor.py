from __future__ import annotations

import re

from .schema import QuoteSpan


SMART_SINGLE_QUOTE_RE = re.compile(r"\u2018(?P<single>[^\u2019\n]{1,1500})\u2019")


# 允许内层一级嵌套引号（如「...以「词语」为标语...」），外层最多 1500 字
BRACKET_QUOTE_RE = re.compile(r"「(?P<bracket>(?:[^「」]|「[^「」]*」){1,1500})」|\"(?P<ascii>[^\"\n]{1,300})\"")
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

# 「直接前跟以下动词时，该引号是名词性/术语引用，不是台词：
#   称为/叫做/叫作/名为/名叫/写着/写有/写道/题为/命名为/取名/起名
_NOMINAL_BEFORE_RE = re.compile(
    r"(?:称为|叫做|叫作|名为|名叫|写着|写有|写道|题为|命名为|取名|起名|简称|自称|"
    r"称之为|统称为|被称为|被叫做|俗称)\s*$"
)


def _window(text: str, start: int, end: int, size: int) -> tuple[str, str]:
    before = text[max(0, start - size):start].strip()
    after = text[end:min(len(text), end + size)].strip()
    return before, after


def _is_nominal_quote(raw_start: int, raw_end: int, full_text: str) -> bool:
    """返回 True 表示该引号是名词/描述性用法，不应提取为台词。

    检查两个方向：
    1. 引号后紧跟「的/这个/那个/一声」等后置助词
    2. 引号前紧跟「称为/叫做/写着」等名词化动词
    """
    # 检查后置
    after = full_text[raw_end:raw_end + 4].lstrip()
    if _NOMINAL_AFTER_RE.match(after):
        return True
    # 检查前置（取引号前 10 字）
    before = full_text[max(0, raw_start - 10):raw_start]
    if _NOMINAL_BEFORE_RE.search(before):
        return True
    return False


def extract_quotes(text: str, context_chars: int = 120, prefix: str = "q") -> list[QuoteSpan]:
    extracted: list[tuple[int, int, int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    for match in BRACKET_QUOTE_RE.finditer(text):
        quote_text = (match.group("bracket") or match.group("ascii") or "").strip().strip("\u2018\u2019")
        # 过滤名词性引用（「X」的声音 / 「X」这个 / 「X」一声 / 称为「X」）
        if _is_nominal_quote(match.start(), match.end(), text):
            continue
        extracted.append((match.start(), match.end(), match.start(), match.end(), quote_text, match.group(0)))
        occupied.append((match.start(), match.end()))

    for match in SMART_SINGLE_QUOTE_RE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        quote_text = (match.group("single") or "").strip()
        if not quote_text or _is_nominal_quote(match.start(), match.end(), text):
            continue
        extracted.append((match.start(), match.end(), match.start("single"), match.end("single"), quote_text, match.group(0)))
        occupied.append((match.start(), match.end()))

    for match in COLON_QUOTE_RE.finditer(text):
        if any(start < match.end("quote") and match.start("quote") < end for start, end in occupied):
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
