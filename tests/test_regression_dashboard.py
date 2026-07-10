from pathlib import Path

from tools.build_regression_dashboard import build_dashboard


def test_dashboard_contains_overall_and_confidence_buckets():
    data = build_dashboard(Path("docs/samples"))
    assert data["overall"]["named"] > 2000
    assert data["overall"]["accuracy"] > 0.8
    assert data["confidence_buckets"]
    assert isinstance(data["error_categories"], dict)
