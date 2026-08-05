from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class Material:
    name: str
    symmetry: Literal["isotropic", "transverse_isotropic"]
    young_modulus: float
    poisson_ratio: float
    transverse_modulus: float | None = None
    shear_modulus: float | None = None
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class MeshBatch:
    nodes: Tensor
    elements: Tensor
    features: Tensor
    forces: Tensor
    constraints: Tensor
    stress: Tensor | None = None
    displacement: Tensor | None = None
    strain: Tensor | None = None
    sample_ids: list[str] = field(default_factory=list)

    def to(self, device: torch.device) -> "MeshBatch":
        return MeshBatch(
            nodes=self.nodes.to(device),
            elements=self.elements.to(device),
            features=self.features.to(device),
            forces=self.forces.to(device),
            constraints=self.constraints.to(device),
            stress=None if self.stress is None else self.stress.to(device),
            displacement=None if self.displacement is None else self.displacement.to(device),
            strain=None if self.strain is None else self.strain.to(device),
            sample_ids=self.sample_ids,
        )

    @property
    def batch_size(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.elements.shape[1])


@dataclass(frozen=True)
class ModelConfig:
    input_dimension: int = 16
    state_dimension: int = 256
    output_dimension: int = 6
    layers: int = 6
    levels: int = 3
    orderings: int = 4
    quadrature_points: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    world_size: int = 4
    physics_weight_start: float = 1.0
    physics_weight_end_epoch: int = 180
    gradient_clip: float | None = None
    precision: str = "fp32"


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class Prediction:
    stress: Tensor
    strain: Tensor
    displacement: Tensor
    hidden: Tensor
    stiffness: Tensor
    transition: Tensor


@dataclass(frozen=True)
class ResidualReport:
    equilibrium: float
    constitutive: float
    compatibility: float
    energy: float
    symmetry: float


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    coverage: float
    width: float
    sharpness: float
