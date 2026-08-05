import torch
from pe_ssm.measurements import coefficient_of_determination, prediction_interval_coverage
from pe_ssm.objectives import physics_weight, relative_l2
from pe_ssm.uncertainty import ConformalCalibrator


def test_exact_prediction_metrics() -> None:
    target = torch.randn(4, 10, 6)
    assert relative_l2(target, target) == 0
    assert torch.allclose(coefficient_of_determination(target, target), torch.tensor(1.0))


def test_physics_annealing() -> None:
    assert physics_weight(0, 180) == 1.0
    assert physics_weight(90, 180) == 0.5
    assert physics_weight(180, 180) == 0.0


def test_conformal_interval_covers_calibration_values() -> None:
    target = torch.linspace(1.0, 10.0, 60).reshape(10, 6)
    prediction = target * 0.95
    calibrator = ConformalCalibrator.fit(prediction, target, alpha=0.1)
    lower, upper = calibrator.interval(prediction)
    assert prediction_interval_coverage(lower, upper, target) >= 0.9
