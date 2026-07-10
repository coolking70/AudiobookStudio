from tools.evaluate_evidence_review import apply_reviews


def test_address_term_revision_requires_explicit_speaker_evidence():
    segments = [{"speaker": "甲", "confidence": 0.6, "evidence": "称呼反推"}]
    reviews = [{
        "index": 0, "decision": "revise", "speaker": "乙", "confidence": 0.95,
        "counter_evidence_type": "address_term", "reason": "乙是称呼对象",
    }]
    updated, stats = apply_reviews(segments, reviews, min_confidence=0.7, review_style="evidence")
    assert updated[0]["speaker"] == "甲"
    assert stats["blocked_by_address_term_gate"] == 1


def test_address_term_can_pass_with_explicit_action_subject():
    segments = [{"speaker": "甲", "confidence": 0.6, "evidence": "称呼反推"}]
    reviews = [{
        "index": 0, "decision": "revise", "speaker": "乙", "confidence": 0.95,
        "counter_evidence_type": "address_term", "reason": "乙先开口说：动作主语明确",
    }]
    updated, stats = apply_reviews(segments, reviews, min_confidence=0.7, review_style="evidence")
    assert updated[0]["speaker"] == "乙"
    assert stats["revised"] == 1
    assert stats["chain_review_indices"] == [0]
