import torch
from pe_ssm.model.operator import PhysicsEncodedStateSpaceModel
from pe_ssm.schema import ModelConfig


def test_model_output_contract() -> None:
    config = ModelConfig(input_dimension=16, state_dimension=32, layers=2, orderings=2)
    model = PhysicsEncodedStateSpaceModel(config)
    features = torch.randn(2, 12, 16)
    orders = torch.stack((torch.arange(12), torch.arange(11, -1, -1)))
    output = model(features, orders)
    assert output.stress.shape == (2, 12, 6)
    assert output.strain.shape == (2, 12, 6)
    assert output.displacement.shape == (2, 12, 3)
    assert output.stiffness.shape == (2, 12, 6, 6)


def test_model_backward_is_finite() -> None:
    config = ModelConfig(input_dimension=16, state_dimension=16, layers=1)
    model = PhysicsEncodedStateSpaceModel(config)
    loss = model(torch.randn(1, 5, 16)).stress.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
