from confidence_calibration import apply_calibration, calibration_metrics, fit_calibrator


def test_fit_calibrator_is_monotonic_and_bounds_values():
    artifact = fit_calibrator([(0.1, False), (0.2, False), (0.8, True), (0.9, True)], bins=4)
    assert artifact["samples"] == 4
    assert all(0.0 <= value <= 1.0 for value in artifact["values"])
    assert artifact["values"] == sorted(artifact["values"])
    assert apply_calibration(0.95, artifact) >= apply_calibration(0.1, artifact)


def test_calibration_metrics_reports_brier_and_ece():
    pairs = [(0.1, False), (0.2, False), (0.8, True), (0.9, True)]
    metrics = calibration_metrics(pairs, bins=4)
    assert metrics["samples"] == 4
    assert metrics["brier"] == 0.025
    assert metrics["ece"] == 0.15
