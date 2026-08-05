# Physics-encoded state space models for additively manufactured implants

PE-SSM predicts displacement, strain, and symmetric stress fields on three-dimensional metallic lattice meshes. The transition operator carries the finite-element stiffness structure formed from the constitutive tensor and strain-displacement matrix, while selective state updates adapt the operator to local geometry. The release includes load-path mesh ordering, multi-order fusion, elasticity operators, conformal intervals, composition-aware splitting, training utilities, and evaluation metrics.

## Installation

Python 3.11 and CUDA 12.4 are the reference environment.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

The Conda environment is installed with `conda env create -f environment.yml`. The container is built with `docker build -t pe-ssm .`.

## Data

Verified public sources are listed in `datasets.txt`. The NIST AM-Bench geometries are public-domain inputs for D1. BAM elastic-property records use CC BY 4.0 and provide calibration values. The generated 7,500-simulation benchmark is not yet assigned a public DOI, so the training arrays themselves are not claimed as downloadable here.

Each processed sample is an HDF5 file containing `nodes`, `elements`, `features`, `forces`, `constraints`, `stress`, `displacement`, and `strain`. File attributes identify topology, alloy, loading, and sample identity. C3D10 meshes should be converted to this schema without changing node units or stress units.

## Configuration

`config/main.yaml` records the reported primary setting: six state-space layers, state dimension 256, four mesh orderings, batch size 32, AdamW at 3e-4 learning rate, 1e-2 weight decay, cosine scheduling, 200 epochs, and four A100 40GB GPUs. The effective global batch is 128. Five seeds are used for reported means and standard deviations.

Ablation files cover removal of equilibrium, constitutive, compatibility, multi-resolution, and multi-order components as well as replacement by a soft physics objective. The experiment registry spans topology, alloy, loading, noise, density, mesh scale, and seed combinations used to organize robustness and transfer evaluations.

## Training

```bash
pe-ssm-train --config config/main.yaml --output runs/main.pt
```

The reported run uses four NVIDIA A100 40GB GPUs and takes about 18.2 hours at state dimension 256. D2 contains 5,000 simulations with a 70/15/15 topology-and-alloy-stratified split. The physics regularizer is annealed to zero before the end of training.

## Evaluation

```bash
pe-ssm-evaluate prediction.pt target.pt
```

The main D2 target is relative L2 stress error 1.41% with a reported standard deviation of 0.09 percentage points over five seeds. Additional targets are stress RMSE 3.52 MPa, displacement MAE 0.0031 mm, equilibrium residual 0.08%, von Mises R² 0.9976, and 90% conformal coverage 90.8%. Numerical comparisons should use the same split, mesh units, material constants, and five seeds.

## Inference

```bash
pe-ssm-infer features.pt prediction.pt --config config/main.yaml
```

Measured single-A100 inference ranges from 0.08 seconds at 10,000 nodes to 8.94 seconds at one million nodes. Peak memory ranges from 0.3GB to 12.3GB over the same range.

## Validation

```bash
pytest -q
ruff check .
mypy --strict code/pe_ssm
```

The tests cover elasticity invariants, tensor shapes, gradient finiteness, loss annealing, conformal coverage, and single-batch overfitting regression.

