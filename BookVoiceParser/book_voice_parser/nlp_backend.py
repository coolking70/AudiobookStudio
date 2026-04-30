from __future__ import annotations

import os
from typing import Any, Protocol


PERSON_LABELS = {"PERSON", "PER", "NR", "Nh", "人名", "PERSON_NAME"}


class NERBackend(Protocol):
    def extract_person_names(self, text: str) -> list[str]:
        ...


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = str(name or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _is_person_label(label: Any) -> bool:
    return str(label or "").strip() in PERSON_LABELS


def _extract_from_tuple(item: tuple[Any, ...]) -> str | None:
    if len(item) < 2:
        return None
    text = item[0]
    if any(_is_person_label(label) for label in item[1:]):
        return str(text or "").strip()
    return None


def _extract_from_dict(item: dict[str, Any]) -> list[str]:
    text = item.get("text") or item.get("word") or item.get("span")
    label = item.get("label") or item.get("type") or item.get("tag")
    if text and _is_person_label(label):
        return [str(text).strip()]

    names: list[str] = []
    for value in item.values():
        names.extend(_extract_person_names(value))
    return names


def _extract_person_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return _extract_from_dict(value)
    if isinstance(value, tuple):
        name = _extract_from_tuple(value)
        return [name] if name else []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            names.extend(_extract_person_names(item))
        return names
    return []


class HanLPNERBackend:
    """Optional HanLP NER adapter for person-name candidate generation."""

    def __init__(self, model_name: str | None = None, tokenizer_model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("BOOK_VOICE_PARSER_HANLP_MODEL", "MSRA_NER_ELECTRA_SMALL_ZH")
        self.tokenizer_model_name = (
            tokenizer_model_name
            or os.getenv("BOOK_VOICE_PARSER_HANLP_TOKENIZER_MODEL", "COARSE_ELECTRA_SMALL_ZH")
        )
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _resolve_model_ref(self, hanlp: Any, namespace: str, model_name: str) -> Any:
        if "." in model_name or "/" in model_name or ":" in model_name:
            return model_name
        try:
            return getattr(getattr(hanlp.pretrained, namespace), model_name)
        except AttributeError as exc:
            raise RuntimeError(f"HanLP {namespace} model not found: {model_name}") from exc

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import hanlp
        except ImportError as exc:
            raise RuntimeError("HanLP is not installed. Install with: pip install 'book-voice-parser[nlp]'") from exc
        self._model = hanlp.load(self._resolve_model_ref(hanlp, "ner", self.model_name))
        return self._model

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import hanlp
        except ImportError as exc:
            raise RuntimeError("HanLP is not installed. Install with: pip install 'book-voice-parser[nlp]'") from exc
        self._tokenizer = hanlp.load(self._resolve_model_ref(hanlp, "tok", self.tokenizer_model_name))
        return self._tokenizer

    def extract_person_names(self, text: str) -> list[str]:
        if not text.strip():
            return []
        tokens = self._load_tokenizer()(text)
        result = self._load()(tokens)
        return _dedupe(_extract_person_names(result))
