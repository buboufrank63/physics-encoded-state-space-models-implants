import argparse
import json
from pathlib import Path

import torch

from pe_ssm.measurements import stress_metrics


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="pe-ssm-evaluate")
    value.add_argument("prediction", type=Path)
    value.add_argument("target", type=Path)
    return value


def main() -> None:
    arguments = parser().parse_args()
    prediction = torch.load(arguments.prediction, map_location="cpu", weights_only=True)
    target = torch.load(arguments.target, map_location="cpu", weights_only=True)
    result = stress_metrics(prediction, target)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
