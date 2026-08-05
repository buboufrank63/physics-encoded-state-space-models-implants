import torch
from torch import Tensor


def relative_l2(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean()


def equilibrium_residual(
    stress_divergence: Tensor, body_force: Tensor, epsilon: float = 1e-8
) -> Tensor:
    residual = stress_divergence + body_force
    numerator = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(body_force.flatten(1), dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean()


def constitutive_residual(
    stress: Tensor, stiffness: Tensor, strain: Tensor, epsilon: float = 1e-8
) -> Tensor:
    expected = torch.matmul(stiffness, strain.unsqueeze(-1)).squeeze(-1)
    numerator = torch.linalg.matrix_norm((stress - expected).flatten(1, -2), ord="fro")
    denominator = torch.linalg.matrix_norm(stress.flatten(1, -2), ord="fro").clamp_min(epsilon)
    return (numerator / denominator).mean()


def compatibility_residual(
    strain: Tensor, symmetric_gradient: Tensor, epsilon: float = 1e-8
) -> Tensor:
    numerator = torch.linalg.vector_norm((strain - symmetric_gradient).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(strain.flatten(1), dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean()


def symmetry_error(stress: Tensor, epsilon: float = 1e-8) -> Tensor:
    difference = stress - stress.transpose(-1, -2)
    numerator = torch.linalg.matrix_norm(difference, ord="fro")
    denominator = torch.linalg.matrix_norm(stress, ord="fro").clamp_min(epsilon)
    return (numerator / denominator).mean()


def physics_weight(epoch: int, end_epoch: int, initial: float = 1.0) -> float:
    if end_epoch <= 0 or epoch >= end_epoch:
        return 0.0
    ratio = 1.0 - float(epoch) / float(end_epoch)
    return initial * ratio


def total_objective(
    stress_prediction: Tensor,
    stress_target: Tensor,
    equilibrium: Tensor,
    constitutive: Tensor,
    compatibility: Tensor,
    weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    data = relative_l2(stress_prediction, stress_target)
    physics = equilibrium + constitutive + compatibility
    total = data + weight * physics
    return total, {
        "total": total.detach(),
        "data": data.detach(),
        "physics": physics.detach(),
        "equilibrium": equilibrium.detach(),
        "constitutive": constitutive.detach(),
        "compatibility": compatibility.detach(),
    }
