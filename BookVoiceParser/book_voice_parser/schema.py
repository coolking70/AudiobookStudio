from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttributionType(str, Enum):
    NARRATOR = "narrator"
    EXPLICIT_BEFORE = "explicit_before"
    EXPLICIT_AFTER = "explicit_after"
    EXPLICIT_NEAR = "explicit_near"
    IMPLICIT = "implicit"
    LATENT = "latent"
    GROUP = "group"
    UNKNOWN = "unknown"


class SegmentEx(BaseModel):
    """AudiobookStudio-compatible segment plus attribution metadata."""

    speaker: str = Field(default="旁白")
    text: str
    emotion: str = Field(default="neutral")
    style: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    voice_engine: str | None = None
    voice_name: str | None = None
    voice_locale: str | None = None
    cfg_value: float | None = None
    inference_timesteps: int | None = None
    addressee: str | None = None
    confidence: float = 1.0
    evidence: str | None = None
    attribution_type: AttributionType | str | None = None
    quote_id: str | None = None
    candidates: list[str] = Field(default_factory=list)
    candidate_sources: dict[str, list[str]] = Field(default_factory=dict)
    scene_characters: list[str] = Field(default_factory=list)
    # 原始上下文窗口，供 review_router 复核时构造 prompt
    context_before: str = ""
    context_after: str = ""


class QuoteSpan(BaseModel):
    quote_id: str
    text: str
    start: int
    end: int
    raw_start: int | None = None
    raw_end: int | None = None
    context_before: str = ""
    context_after: str = ""
    raw: str = ""


class CandidateSet(BaseModel):
    quote_id: str
    candidates: list[str] = Field(default_factory=list)
    candidate_sources: dict[str, list[str]] = Field(default_factory=dict)
    scene_characters: list[str] = Field(default_factory=list)


class Attribution(BaseModel):
    quote_id: str
    speaker: str
    confidence: float
    evidence: str | None = None
    attribution_type: AttributionType | str = AttributionType.UNKNOWN
    addressee: str | None = None


class ParseResult(BaseModel):
    segments: list[SegmentEx] = Field(default_factory=list)
    review_items: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
