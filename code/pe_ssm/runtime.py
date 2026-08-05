import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pe_ssm.model.operator import PhysicsEncodedStateSpaceModel
from pe_ssm.objectives import relative_l2
from pe_ssm.schema import ExperimentConfig, ModelConfig, TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> ExperimentConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = ModelConfig(**values.get("model", {}))
    train_values = dict(values.get("train", {}))
    train_values["batch_size"] = values.get("data", {}).get("batch_size", 32)
    train = TrainConfig(
        **{
            key: value
            for key, value in train_values.items()
            if key in TrainConfig.__dataclass_fields__
        }
    )
    return ExperimentConfig(seed=int(values.get("seed", 1)), model=model, train=train)


def atomic_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    seed = int(state["seed"])
    set_seed(seed)
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return int(state["epoch"]), seed


class Trainer:
    def __init__(
        self, model: PhysicsEncodedStateSpaceModel, config: ExperimentConfig, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.train.epochs)

    def train_step(self, features: Tensor, target: Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(features.to(self.device))
        loss = relative_l2(prediction.stress, target.to(self.device))
        loss.backward()
        if self.config.train.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip)
        self.optimizer.step()
        return {"loss": float(loss.detach())}

    @torch.no_grad()
    def evaluate_step(self, features: Tensor, target: Tensor) -> dict[str, float]:
        self.model.eval()
        prediction = self.model(features.to(self.device))
        loss = relative_l2(prediction.stress, target.to(self.device))
        return {"relative_l2": float(loss)}

    def finish_epoch(self) -> None:
        self.scheduler.step()

    def save(self, path: Path, epoch: int) -> None:
        atomic_checkpoint(path, self.model, self.optimizer, self.scheduler, epoch, self.config.seed)

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
