import argparse
import logging
from pathlib import Path

import torch

from pe_ssm.model.operator import PhysicsEncodedStateSpaceModel
from pe_ssm.runtime import Trainer, load_config, set_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="pe-ssm-train")
    value.add_argument("--config", type=Path, default=Path("config/main.yaml"))
    value.add_argument("--output", type=Path, default=Path("runs/main.pt"))
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(arguments.config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsEncodedStateSpaceModel(config.model)
    trainer = Trainer(model, config, device)
    logging.getLogger(__name__).info("model_parameters=%d", model.parameter_count())
    trainer.save(arguments.output, 0)


if __name__ == "__main__":
    main()
