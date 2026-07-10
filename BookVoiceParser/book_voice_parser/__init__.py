"""Chinese-BookNLP-Lite public API."""

from .batch_llm_attributor import BatchConfig, BatchLLMAttributor
from .parser import parse_novel
from .review_router import LLMRouterConfig, route_dense_to_llm, route_to_batch_llm, route_to_llm
from .schema import AttributionType, ParseResult, QuoteSpan, SegmentEx
from .spc_ranker import OpenAICompatibleSPCRanker

__all__ = [
    "AttributionType",
    "BatchConfig",
    "BatchLLMAttributor",
    "LLMRouterConfig",
    "OpenAICompatibleSPCRanker",
    "ParseResult",
    "QuoteSpan",
    "SegmentEx",
    "parse_novel",
    "route_to_batch_llm",
    "route_to_llm",
    "route_dense_to_llm",
]

