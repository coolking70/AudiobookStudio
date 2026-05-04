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


# \u7535\u5b50\u4e66\u7ad9\u70b9\u6c34\u5370\u884c\uff08\u5e38\u89c1\u4e8e\u8f6c\u8f7d\u8f7b\u5c0f\u8bf4\u6587\u672c\uff09
# \u5339\u914d"\u2605\u2606\u2605" / "\u3010\u8f7b\u5c0f\u8bf4...\u3011" / "www.xxx.com" \u7b49\u6a21\u5f0f\u7684\u72ec\u7acb\u884c
#
# \u4fdd\u5b88\u7b56\u7565\uff1a\u6c34\u5370\u6807\u8bb0\u5fc5\u987b\u51fa\u73b0\u5728\u884c\u9996\uff08\u6700\u591a\u5141\u8bb8 8 \u4e2a\u7a7a\u683c/\u5236\u8868\u7b26\u4f5c\u4e3a\u7f29\u8fdb\u524d\u7f00\uff09\u3002
# \u8fd9\u6837\u5373\u4fbf\u884c\u5185\u5b58\u5728\u6c34\u5370\u5173\u952e\u5b57\uff0c\u53ea\u8981\u524d\u9762\u6709\u6b63\u6587\u5185\u5bb9\uff08\u5982"\u4ed6\u6253\u5f00\u4e86www.xxx.com"\uff09\uff0c
# \u6574\u884c\u5c31\u4e0d\u4f1a\u88ab\u8bef\u5220\u3002
_EBOOK_WATERMARK_RE = re.compile(
    r"(?:^|\n)"                           # \u884c\u9996\u6216\u6bb5\u9996
    r"[ \t]{0,8}"                         # \u4ec5\u5141\u8bb8\u884c\u9996\u7f29\u8fdb\u7a7a\u767d\uff0c\u4e0d\u5141\u8bb8\u6b63\u6587\u5185\u5bb9\u51fa\u73b0\u5728\u6807\u8bb0\u524d
    r"(?:"
    r"\u2605[\u2606\u2605]{2,}"                         # \u2605\u2606\u2605\u2606... \u88c5\u9970\u7b26
    r"|\u3010[^\u3011]{0,30}(?:\u6587\u5e93|\u5c0f\u8bf4|\u8f7b\u5c0f\u8bf4|Light\s*Novel)[^\u3011]{0,30}\u3011"
    r"|Www?\.[A-Za-z0-9]+\.[A-Za-z]{2,5}"  # \u7f51\u7ad9\u57df\u540d\uff08\u884c\u9996\u624d\u5220\uff09
    r"|\u626b\u56fe[:\uff1a]|\u5f55\u5165[:\uff1a]|\u4fee\u56fe[:\uff1a]|\u56fe\u6e90[:\uff1a]|\u8f6c\u81ea[:\uff1a]"  # \u626b\u56fe/\u5f55\u5165\u5143\u4fe1\u606f
    r")"
    r"[^\n]*",
    re.IGNORECASE,
)

# \u8f7b\u5c0f\u8bf4\u6587\u5e93 / \u8f7b\u4e4b\u56fd\u5ea6 \u7b49\u7ad9\u70b9\u7279\u5f81\u6027\u524d\u7f00\u5757\uff08\u6574\u5757\u6e05\u9664\uff0c\u907f\u514d\u53ea\u5220\u4e2d\u95f4\u7559\u7a7a\uff09
#
# \u4fdd\u5b88\u7b56\u7565\uff1a\u82e5\u67d0\u884c\u5728\u7ad9\u70b9\u5173\u952e\u5b57\u51fa\u73b0\u4e4b\u524d\u542b\u6709\u53e5\u672b\u6807\u70b9\uff08\u3002\uff01\uff1f\uff09\uff0c
# \u8bf4\u660e\u8be5\u884c\u542b\u6709\u6b63\u6587\u53d9\u8ff0\u5185\u5bb9\u800c\u975e\u7eaf\u6c34\u5370\u884c\uff0c\u4e0d\u4e88\u5339\u914d\uff0c\u9632\u6b62\u8bef\u5220\u3002
_EBOOK_HEADER_RE = re.compile(
    r"^(?:(?![^\n]*[\u3002\uff01\uff1f])[^\n]*(?:\u8f7b\u5c0f\u8bf4\u6587\u5e93|\u8f7b\u4e4b\u56fd\u5ea6|\u5929\u4f7f\u52a8\u6f2b|WenKu8|wenku8|linovel)[^\n]*\n?)+"
    r"(?:(?![^\n]*[\u3002\uff01\uff1f])[^\n]*(?:\u626b\u56fe|\u5f55\u5165|\u4fee\u56fe|\u56fe\u6e90|\u53f0\u7248|\u7b80\u4f53|\u7e41\u4f53|\u8f6c\u81ea)[^\n]*\n?)*",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace while preserving paragraph boundaries."""

    normalized = (text or "").replace("\ufeff", "").translate(QUOTE_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # \u6e05\u9664\u7535\u5b50\u4e66\u6c34\u5370/\u7248\u6743\u5934\uff08\u5fc5\u987b\u5728\u5176\u4ed6\u5904\u7406\u4e4b\u524d\uff0c\u907f\u514d\u6c34\u5370\u62c6\u6563\u8fdb\u6bb5\u843d\uff09
    normalized = _EBOOK_HEADER_RE.sub("", normalized)
    normalized = _EBOOK_WATERMARK_RE.sub("", normalized)

    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_chapters(text: str) -> list[tuple[str, str]]:
    """Split text into rough chapters. Falls back to a single full-text chapter."""

    normalized = normalize_text(text)
    if not normalized:
        return []

    pattern = re.compile(r"(?m)^\s*(第[零一二三四五六七八九十百千万\d]+[章节回卷].*|Chapter\s+\d+.*)\s*$")
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

