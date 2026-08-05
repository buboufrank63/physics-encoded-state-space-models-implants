import argparse
from pathlib import Path

import torch

from pe_ssm.model.operator import PhysicsEncodedStateSpaceModel
from pe_ssm.runtime import load_config


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="pe-ssm-infer")
    value.add_argument("features", type=Path)
    value.add_argument("output", type=Path)
    value.add_argument("--config", type=Path, default=Path("config/main.yaml"))
    return value


def main() -> None:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    model = PhysicsEncodedStateSpaceModel(config.model).eval()
    features = torch.load(arguments.features, map_location="cpu", weights_only=True)
    with torch.no_grad():
        prediction = model(features)
    torch.save(
        {
            "stress": prediction.stress,
            "strain": prediction.strain,
            "displacement": prediction.displacement,
        },
        arguments.output,
    )


if __name__ == "__main__":
    main()
