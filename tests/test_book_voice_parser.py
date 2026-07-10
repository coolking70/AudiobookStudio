from BookVoiceParser.book_voice_parser.quote_extractor import extract_quotes
from BookVoiceParser.book_voice_parser.audit import make_audit_prompt, select_production_targets
from BookVoiceParser.book_voice_parser.review_router import (
    _should_auto_apply_review,
    route_dense_to_llm,
    route_to_batch_llm,
)
from BookVoiceParser.book_voice_parser.schema import Attribution, SegmentEx


def test_quote_extractor_keeps_order_and_context():
    quotes = extract_quotes("他停下来说：‘你好。’她回答：‘再见。’")
    assert [q.text for q in quotes] == ["你好。", "再见。"]
    assert quotes[0].quote_id != quotes[1].quote_id


def test_dense_route_skips_sparse_dialogue_without_model_call():
    segments = [
        SegmentEx(quote_id="q1", text="甲", speaker="甲", confidence=0.9),
        SegmentEx(quote_id="q2", text="乙", speaker="乙", confidence=0.9),
    ]
    routed, stats = route_dense_to_llm(segments, object())
    assert [item.speaker for item in routed] == ["甲", "乙"]
    assert stats["targets"] == 0


def test_production_target_selection_requires_signal_intersection():
    segments = [
        {"speaker": "甲", "text": "a", "confidence": 0.95, "candidates": ["甲"]},
        {"speaker": "乙", "text": "b", "confidence": 0.95, "candidates": ["乙"]},
        {"speaker": "丙", "text": "c", "confidence": 0.95, "candidates": ["丙"]},
        {"speaker": "甲", "text": "d", "confidence": 0.4, "candidates": ["乙", "甲"]},
    ]
    targets, reasons, counts = select_production_targets(segments)
    assert targets == [3]
    assert set(reasons[3]) >= {"low_confidence", "dense_scene", "candidate_conflict"}
    assert counts["selected"] == 1


def test_production_target_selection_includes_unresolved_speaker():
    segments = [
        {"speaker": "未知", "text": "a", "confidence": 0.4, "candidates": []},
        {"speaker": "旁白", "text": "n", "confidence": 0.1, "candidates": []},
    ]
    targets, reasons, counts = select_production_targets(segments)
    assert targets == [0]
    assert set(reasons[0]) >= {"low_confidence", "unresolved"}
    assert counts["unresolved"] == 1


def test_online_review_blocks_address_term_without_explicit_speaker_evidence():
    segment = SegmentEx(
        quote_id="q1",
        text="姐姐，你来了。",
        speaker="甲",
        confidence=0.6,
        candidates=["甲", "乙"],
        candidate_sources={"乙": ["address_term_backcheck"]},
    )
    allowed, reason = _should_auto_apply_review(segment, "乙", 0.95, "乙是被称呼的姐姐")
    assert allowed is False
    assert "称呼关系" in reason


def test_online_review_allows_address_term_with_explicit_action_subject():
    segment = SegmentEx(
        quote_id="q1",
        text="姐姐，你来了。",
        speaker="甲",
        confidence=0.6,
        candidates=["甲", "乙"],
        candidate_sources={"乙": ["address_term_backcheck"]},
    )
    allowed, reason = _should_auto_apply_review(segment, "乙", 0.95, "后文明确乙开口回答")
    assert allowed is True
    assert reason == ""


def test_batch_review_rechecks_neighbors_after_correction(monkeypatch):
    calls: list[list[str]] = []

    def fake_attribute_batch(self, batch, **_kwargs):
        quote_ids = [quote.quote_id for quote, _ in batch]
        calls.append(quote_ids)
        if quote_ids == ["q1"]:
            return [Attribution(
                quote_id="q1",
                speaker="乙",
                confidence=0.95,
                evidence="后文明确乙开口回答",
            )]
        return [Attribution(
            quote_id=quote_id,
            speaker="甲",
            confidence=0.95,
            evidence="前文明确甲说道",
        ) for quote_id in quote_ids]

    monkeypatch.setattr(
        "BookVoiceParser.book_voice_parser.review_router.BatchLLMAttributor.attribute_batch",
        fake_attribute_batch,
    )
    segments = [
        SegmentEx(quote_id="q0", text="a", speaker="甲", confidence=0.9),
        SegmentEx(
            quote_id="q1",
            text="b",
            speaker="甲",
            confidence=0.4,
            candidates=["甲", "乙"],
            candidate_sources={"乙": ["role_hints"]},
        ),
        SegmentEx(quote_id="q2", text="c", speaker="甲", confidence=0.9),
    ]

    updated, stats = route_to_batch_llm(
        segments,
        object(),
        review_indices=[1],
    )

    assert updated[1].speaker == "乙"
    assert calls == [["q1"], ["q0", "q2"]]
    assert stats["chain_review_indices"] == [0, 2]
    assert stats["reviewed"] == 3


def test_audit_prompt_requests_audit_safe_fields():
    prompt = make_audit_prompt("甲、乙", None, "前文", "你好", "后文")
    assert "counter_evidence_type" in prompt
    assert "baseline_evidence_valid" in prompt
    assert "auto_apply_safe" in prompt
    assert "仅有称呼关系" in prompt
