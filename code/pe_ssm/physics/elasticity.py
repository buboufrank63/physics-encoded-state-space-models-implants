import torch
from torch import Tensor
from torch.nn import functional as F


def lame_parameters(young_modulus: Tensor, poisson_ratio: Tensor) -> tuple[Tensor, Tensor]:
    denominator = (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    lam = young_modulus * poisson_ratio / denominator
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    return lam, mu


def isotropic_stiffness(young_modulus: Tensor, poisson_ratio: Tensor) -> Tensor:
    lam, mu = lame_parameters(young_modulus, poisson_ratio)
    shape = torch.broadcast_shapes(lam.shape, mu.shape)
    output = torch.zeros(*shape, 6, 6, dtype=lam.dtype, device=lam.device)
    diagonal = lam + 2.0 * mu
    output[..., 0, 0] = diagonal
    output[..., 1, 1] = diagonal
    output[..., 2, 2] = diagonal
    output[..., 0, 1] = lam
    output[..., 0, 2] = lam
    output[..., 1, 0] = lam
    output[..., 1, 2] = lam
    output[..., 2, 0] = lam
    output[..., 2, 1] = lam
    output[..., 3, 3] = mu
    output[..., 4, 4] = mu
    output[..., 5, 5] = mu
    return output


def positive_material_parameters(raw: Tensor, floor: float = 1e-6) -> Tensor:
    return F.softplus(raw) + floor


def transverse_isotropic_stiffness(parameters: Tensor) -> Tensor:
    values = positive_material_parameters(parameters)
    longitudinal = values[..., 0]
    transverse = values[..., 1]
    poisson_lt = torch.sigmoid(values[..., 2]) * 0.49
    poisson_tt = torch.sigmoid(values[..., 3]) * 0.49
    shear_lt = values[..., 4]
    shear_tt = transverse / (2.0 * (1.0 + poisson_tt))
    compliance = torch.zeros(*values.shape[:-1], 6, 6, dtype=values.dtype, device=values.device)
    compliance[..., 0, 0] = 1.0 / transverse
    compliance[..., 1, 1] = 1.0 / transverse
    compliance[..., 2, 2] = 1.0 / longitudinal
    compliance[..., 0, 1] = -poisson_tt / transverse
    compliance[..., 1, 0] = compliance[..., 0, 1]
    compliance[..., 0, 2] = -poisson_lt / longitudinal
    compliance[..., 2, 0] = compliance[..., 0, 2]
    compliance[..., 1, 2] = -poisson_lt / longitudinal
    compliance[..., 2, 1] = compliance[..., 1, 2]
    compliance[..., 3, 3] = 1.0 / shear_lt
    compliance[..., 4, 4] = 1.0 / shear_lt
    compliance[..., 5, 5] = 1.0 / shear_tt
    return torch.linalg.inv(compliance)


def voigt_to_tensor(stress: Tensor) -> Tensor:
    output = torch.zeros(*stress.shape[:-1], 3, 3, dtype=stress.dtype, device=stress.device)
    output[..., 0, 0] = stress[..., 0]
    output[..., 1, 1] = stress[..., 1]
    output[..., 2, 2] = stress[..., 2]
    output[..., 0, 1] = stress[..., 3]
    output[..., 1, 0] = stress[..., 3]
    output[..., 1, 2] = stress[..., 4]
    output[..., 2, 1] = stress[..., 4]
    output[..., 0, 2] = stress[..., 5]
    output[..., 2, 0] = stress[..., 5]
    return output


def tensor_to_voigt(stress: Tensor) -> Tensor:
    return torch.stack(
        (
            stress[..., 0, 0],
            stress[..., 1, 1],
            stress[..., 2, 2],
            0.5 * (stress[..., 0, 1] + stress[..., 1, 0]),
            0.5 * (stress[..., 1, 2] + stress[..., 2, 1]),
            0.5 * (stress[..., 0, 2] + stress[..., 2, 0]),
        ),
        dim=-1,
    )


def symmetrize_stress(stress: Tensor) -> Tensor:
    matrix = voigt_to_tensor(stress) if stress.shape[-1] == 6 else stress
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def von_mises(stress: Tensor) -> Tensor:
    matrix = voigt_to_tensor(stress) if stress.shape[-1] == 6 else stress
    mean = torch.diagonal(matrix, dim1=-2, dim2=-1).mean(dim=-1)
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    deviator = matrix - mean[..., None, None] * identity
    return torch.sqrt(1.5 * torch.sum(deviator * deviator, dim=(-2, -1)).clamp_min(0.0))


def tetrahedron_gradients(coordinates: Tensor) -> tuple[Tensor, Tensor]:
    if coordinates.shape[-2:] != (4, 3):
        raise ValueError("linear tetrahedra require four three-dimensional coordinates")
    ones = torch.ones(
        *coordinates.shape[:-2], 4, 1, dtype=coordinates.dtype, device=coordinates.device
    )
    system = torch.cat((ones, coordinates), dim=-1)
    determinant = torch.linalg.det(system)
    volume = determinant.abs() / 6.0
    inverse = torch.linalg.inv(system)
    gradients = inverse[..., 1:, :].transpose(-1, -2)
    return gradients, volume


def strain_displacement_matrix(gradients: Tensor) -> Tensor:
    node_count = gradients.shape[-2]
    output = torch.zeros(
        *gradients.shape[:-2],
        6,
        node_count * 3,
        dtype=gradients.dtype,
        device=gradients.device,
    )
    for node in range(node_count):
        x = gradients[..., node, 0]
        y = gradients[..., node, 1]
        z = gradients[..., node, 2]
        column = node * 3
        output[..., 0, column] = x
        output[..., 1, column + 1] = y
        output[..., 2, column + 2] = z
        output[..., 3, column] = y
        output[..., 3, column + 1] = x
        output[..., 4, column + 1] = z
        output[..., 4, column + 2] = y
        output[..., 5, column] = z
        output[..., 5, column + 2] = x
    return output


def assemble_element_stiffness(b_matrix: Tensor, constitutive: Tensor, volume: Tensor) -> Tensor:
    stiffness = b_matrix.transpose(-1, -2) @ constitutive @ b_matrix
    return stiffness * volume[..., None, None]


def lumped_mass_matrix(volume: Tensor, degrees: int, density: Tensor | float = 1.0) -> Tensor:
    density_tensor = torch.as_tensor(density, dtype=volume.dtype, device=volume.device)
    mass = volume * density_tensor
    diagonal = mass[..., None].expand(*mass.shape, degrees) / float(degrees)
    return torch.diag_embed(diagonal)


def stable_transition(stiffness: Tensor, mass: Tensor, step: Tensor) -> Tensor:
    diagonal = torch.diagonal(mass, dim1=-2, dim2=-1).clamp_min(1e-8)
    generator = -stiffness / diagonal[..., :, None]
    return torch.matrix_exp(generator * step[..., None, None])


def strain_from_displacement(b_matrix: Tensor, displacement: Tensor) -> Tensor:
    return torch.matmul(b_matrix, displacement.unsqueeze(-1)).squeeze(-1)


def stress_from_strain(constitutive: Tensor, strain: Tensor) -> Tensor:
    return torch.matmul(constitutive, strain.unsqueeze(-1)).squeeze(-1)


def strain_energy(stress: Tensor, strain: Tensor, volume: Tensor) -> Tensor:
    return 0.5 * torch.sum(stress * strain, dim=-1) * volume


def rotate_stiffness(constitutive: Tensor, rotation: Tensor) -> Tensor:
    basis = _voigt_rotation(rotation)
    return basis @ constitutive @ basis.transpose(-1, -2)


def _voigt_rotation(rotation: Tensor) -> Tensor:
    columns = []
    dtype = rotation.dtype
    device = rotation.device
    for index in range(6):
        unit = torch.zeros(6, dtype=dtype, device=device)
        unit[index] = 1.0
        tensor = voigt_to_tensor(unit)
        transformed = rotation @ tensor @ rotation.transpose(-1, -2)
        columns.append(tensor_to_voigt(transformed))
    return torch.stack(columns, dim=-1)


def build_direction_rotation(direction: Tensor) -> Tensor:
    normalized = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    reference_x = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
    reference_y = torch.tensor([0.0, 1.0, 0.0], dtype=direction.dtype, device=direction.device)
    use_y = normalized[..., 0].abs() > 0.9
    reference = torch.where(use_y[..., None], reference_y, reference_x)
    tangent = torch.linalg.cross(reference.expand_as(normalized), normalized, dim=-1)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)
    bitangent = torch.linalg.cross(normalized, tangent, dim=-1)
    return torch.stack((tangent, bitangent, normalized), dim=-1)
