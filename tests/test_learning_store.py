import learning_store
import pytest


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


def test_silver_quality_gate_and_version_bundle():
    records = [{
        "version": 1,
        "id": "record-1",
        "text_sha256": "abc",
        "text": "你好",
        "previous_speaker": "甲",
        "speaker": "乙",
    }]
    quality = learning_store.audit_records(records)
    assert quality["valid"] is True
    bundle = learning_store.build_version_bundle(records, min_records=1)
    assert bundle["dataset_id"].startswith("silver-")
    assert bundle["record_count"] == 1
    assert len(bundle["sha256"]) == 64


def test_silver_version_rejects_invalid_or_insufficient_records():
    invalid = [{"id": "x", "text": "", "speaker": "甲", "previous_speaker": "甲"}]
    with pytest.raises(ValueError, match="quality gate"):
        learning_store.build_version_bundle(invalid, min_records=1)
    with pytest.raises(ValueError, match="at least 2"):
        learning_store.build_version_bundle([], min_records=2)
