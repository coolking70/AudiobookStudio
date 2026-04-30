from __future__ import annotations

import re


QUOTE_TRANSLATION = str.maketrans(
    {
        "“": "「",
        "”": "」",
        "『": "「",
        "』": "」",
        "《": "「",
        "》": "」",
    }
)


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace while preserving paragraph boundaries."""

    normalized = (text or "").replace("\ufeff", "").translate(QUOTE_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_chapters(text: str) -> list[tuple[str, str]]:
    """Split text into rough chapters. Falls back to a single full-text chapter."""

    normalized = normalize_text(text)
    if not normalized:
        return []

    pattern = re.compile(r"(?m)^(第[零一二三四五六七八九十百千万\d]+[章节回卷].*|Chapter\s+\d+.*)$")
    matches = list(pattern.finditer(normalized))
    if not matches:
        return [("全文", normalized)]

    chapters: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        content = normalized[start:end].strip()
        if content:
            chapters.append((title, content))
    return chapters or [("全文", normalized)]

