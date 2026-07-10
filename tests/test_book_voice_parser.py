from BookVoiceParser.book_voice_parser.quote_extractor import extract_quotes
from BookVoiceParser.book_voice_parser.review_router import route_dense_to_llm
from BookVoiceParser.book_voice_parser.audit import select_production_targets
from BookVoiceParser.book_voice_parser.schema import SegmentEx


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
