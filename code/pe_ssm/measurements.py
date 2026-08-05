import math

import numpy as np
import torch
from torch import Tensor

from pe_ssm.physics.elasticity import von_mises


def root_mean_square_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(prediction - target))


def coefficient_of_determination(prediction: Tensor, target: Tensor) -> Tensor:
    residual = torch.sum((target - prediction) ** 2)
    centered = torch.sum((target - target.mean()) ** 2).clamp_min(1e-12)
    return 1.0 - residual / centered


def peak_relative_error(prediction: Tensor, target: Tensor) -> Tensor:
    predicted_peak = prediction.abs().amax(dim=-2)
    target_peak = target.abs().amax(dim=-2)
    return torch.mean(torch.abs(predicted_peak - target_peak) / target_peak.clamp_min(1e-8))


def stress_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    predicted_vm = von_mises(prediction)
    target_vm = von_mises(target)
    return {
        "rmse": float(root_mean_square_error(prediction, target)),
        "mae": float(mean_absolute_error(prediction, target)),
        "r2_von_mises": float(coefficient_of_determination(predicted_vm, target_vm)),
        "peak_relative_error": float(peak_relative_error(prediction, target)),
    }


def bootstrap_interval(
    values: np.ndarray,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 1,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def paired_cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    differences = first - second
    deviation = differences.std(ddof=1)
    if deviation == 0.0:
        return math.inf if differences.mean() != 0.0 else 0.0
    return float(differences.mean() / deviation)


def expected_calibration_error(nominal: Tensor, observed: Tensor) -> Tensor:
    if nominal.shape != observed.shape:
        raise ValueError("nominal and observed coverage arrays must match")
    return torch.mean(torch.abs(nominal - observed))


def prediction_interval_coverage(lower: Tensor, upper: Tensor, target: Tensor) -> Tensor:
    contained = (target >= lower) & (target <= upper)
    return contained.float().mean()


def mean_prediction_interval_width(
    lower: Tensor, upper: Tensor, scale: Tensor | None = None
) -> Tensor:
    width = upper - lower
    if scale is not None:
        width = width / scale.abs().clamp_min(1e-8)
    return width.mean()


def linear_scaling_exponent(node_counts: Tensor, durations: Tensor) -> Tensor:
    x = torch.log(node_counts.float())
    y = torch.log(durations.float())
    centered_x = x - x.mean()
    return torch.sum(centered_x * (y - y.mean())) / torch.sum(centered_x**2)
