from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConformalCalibrator:
    quantile: Tensor
    alpha: float
    reference_scale: Tensor

    @classmethod
    def fit(cls, prediction: Tensor, target: Tensor, alpha: float = 0.1) -> "ConformalCalibrator":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        flat_prediction = prediction.flatten(1)
        flat_target = target.flatten(1)
        score = torch.amax(torch.abs(flat_prediction - flat_target), dim=1)
        scale = torch.amax(torch.abs(flat_target), dim=1).clamp_min(1e-8)
        normalized = score / scale
        count = normalized.numel()
        rank = min(count, int(torch.ceil(torch.tensor((count + 1) * (1.0 - alpha))).item()))
        quantile = torch.sort(normalized).values[rank - 1]
        reference_scale = torch.amax(torch.abs(flat_target))
        return cls(quantile=quantile, alpha=alpha, reference_scale=reference_scale)

    def interval(self, prediction: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
        magnitude = self.reference_scale.to(prediction.device) if scale is None else scale
        radius = self.quantile.to(prediction.device) * magnitude
        return prediction - radius, prediction + radius


def coverage_curve(
    prediction: Tensor,
    target: Tensor,
    levels: Tensor,
) -> tuple[Tensor, Tensor]:
    observed = []
    widths = []
    for level in levels:
        calibrator = ConformalCalibrator.fit(prediction, target, alpha=1.0 - float(level))
        lower, upper = calibrator.interval(prediction)
        observed.append(((target >= lower) & (target <= upper)).float().mean())
        widths.append((upper - lower).mean())
    return torch.stack(observed), torch.stack(widths)
