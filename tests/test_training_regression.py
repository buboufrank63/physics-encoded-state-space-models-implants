import torch
from pe_ssm.model.operator import PhysicsEncodedStateSpaceModel
from pe_ssm.runtime import Trainer, set_seed
from pe_ssm.schema import ExperimentConfig, ModelConfig, TrainConfig


def test_single_batch_overfit_regression() -> None:
    set_seed(7)
    config = ExperimentConfig(
        seed=7,
        model=ModelConfig(input_dimension=4, state_dimension=8, layers=1),
        train=TrainConfig(
            epochs=20, learning_rate=0.01, weight_decay=0.0, batch_size=2, world_size=1
        ),
    )
    model = PhysicsEncodedStateSpaceModel(config.model)
    trainer = Trainer(model, config, torch.device("cpu"))
    features = torch.randn(2, 4, 4)
    target = torch.randn(2, 4, 6)
    losses = [trainer.train_step(features, target)["loss"] for _ in range(12)]
    assert min(losses[-4:]) < losses[0]
