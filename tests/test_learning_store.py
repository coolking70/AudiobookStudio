import learning_store


def test_silver_store_deduplicates_corrections(tmp_path, monkeypatch):
    path = tmp_path / "silver.jsonl"
    monkeypatch.setattr(learning_store, "SILVER_PATH", path)
    monkeypatch.setattr(learning_store, "SILVER_DIR", tmp_path)
    segment = {"text": "你好", "speaker": "乙", "confidence": 0.4}
    first = learning_store.append_correction(text="原文", segment=segment, index=2, previous_speaker="甲")
    second = learning_store.append_correction(text="原文", segment=segment, index=2, previous_speaker="甲")
    assert first["stored"] is True
    assert second["reason"] == "duplicate"
    assert learning_store.stats()["count"] == 1
