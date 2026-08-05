import torch
from pe_ssm.physics.elasticity import (
    assemble_element_stiffness,
    isotropic_stiffness,
    strain_displacement_matrix,
    tetrahedron_gradients,
    von_mises,
)


def test_isotropic_tensor_is_symmetric_positive_definite() -> None:
    tensor = isotropic_stiffness(torch.tensor(120000.0), torch.tensor(0.34))
    assert torch.allclose(tensor, tensor.T)
    assert torch.linalg.eigvalsh(tensor).amin() > 0


def test_tetrahedron_volume_and_stiffness() -> None:
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gradients, volume = tetrahedron_gradients(coordinates)
    b_matrix = strain_displacement_matrix(gradients)
    constitutive = isotropic_stiffness(torch.tensor(120000.0), torch.tensor(0.34))
    stiffness = assemble_element_stiffness(b_matrix, constitutive, volume)
    assert torch.allclose(volume, torch.tensor(1.0 / 6.0))
    assert stiffness.shape == (12, 12)
    assert torch.allclose(stiffness, stiffness.T, atol=1e-4)


def test_uniaxial_von_mises() -> None:
    stress = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(von_mises(stress), torch.tensor([100.0]))
