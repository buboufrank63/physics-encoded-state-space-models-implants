from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pe_ssm.schema import MeshBatch


@dataclass(frozen=True)
class SampleIndex:
    path: Path
    topology: str
    alloy: str
    loading: str
    identity: str


class LatticeDataset(Dataset[MeshBatch]):
    def __init__(self, samples: Sequence[SampleIndex], feature_dimension: int = 16) -> None:
        self.samples = list(samples)
        self.feature_dimension = feature_dimension

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MeshBatch:
        sample = self.samples[index]
        with h5py.File(sample.path, "r") as handle:
            nodes = torch.from_numpy(np.asarray(handle["nodes"], dtype=np.float32))
            elements = torch.from_numpy(np.asarray(handle["elements"], dtype=np.int64))
            features = torch.from_numpy(np.asarray(handle["features"], dtype=np.float32))
            forces = torch.from_numpy(np.asarray(handle["forces"], dtype=np.float32))
            constraints = torch.from_numpy(np.asarray(handle["constraints"], dtype=np.bool_))
            stress = torch.from_numpy(np.asarray(handle["stress"], dtype=np.float32))
            displacement = torch.from_numpy(np.asarray(handle["displacement"], dtype=np.float32))
            strain = torch.from_numpy(np.asarray(handle["strain"], dtype=np.float32))
        if features.shape[-1] != self.feature_dimension:
            raise ValueError(
                f"expected {self.feature_dimension} features, received {features.shape[-1]}"
            )
        return MeshBatch(
            nodes=nodes.unsqueeze(0),
            elements=elements.unsqueeze(0),
            features=features.unsqueeze(0),
            forces=forces.unsqueeze(0),
            constraints=constraints.unsqueeze(0),
            stress=stress.unsqueeze(0),
            displacement=displacement.unsqueeze(0),
            strain=strain.unsqueeze(0),
            sample_ids=[sample.identity],
        )


def discover_samples(root: Path) -> list[SampleIndex]:
    samples = []
    for path in sorted(root.glob("**/*.h5")):
        with h5py.File(path, "r") as handle:
            topology = str(handle.attrs["topology"])
            alloy = str(handle.attrs["alloy"])
            loading = str(handle.attrs["loading"])
            identity = str(handle.attrs.get("identity", path.stem))
        samples.append(SampleIndex(path, topology, alloy, loading, identity))
    return samples


def stratified_split(
    samples: Sequence[SampleIndex],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 1,
) -> tuple[list[SampleIndex], list[SampleIndex], list[SampleIndex]]:
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to one")
    groups: dict[tuple[str, str], list[SampleIndex]] = {}
    for sample in samples:
        groups.setdefault((sample.topology, sample.alloy), []).append(sample)
    generator = np.random.default_rng(seed)
    partitions: tuple[list[SampleIndex], list[SampleIndex], list[SampleIndex]] = ([], [], [])
    for group in groups.values():
        shuffled = list(group)
        generator.shuffle(shuffled)
        train_end = int(len(shuffled) * fractions[0])
        validation_end = train_end + int(len(shuffled) * fractions[1])
        partitions[0].extend(shuffled[:train_end])
        partitions[1].extend(shuffled[train_end:validation_end])
        partitions[2].extend(shuffled[validation_end:])
    return partitions


def collate_meshes(samples: Sequence[MeshBatch]) -> MeshBatch:
    if not samples:
        raise ValueError("cannot collate an empty sequence")
    element_counts = {sample.element_count for sample in samples}
    if len(element_counts) != 1:
        raise ValueError("batch collation requires equal element counts")
    optional_names = ("stress", "displacement", "strain")
    optional: dict[str, Tensor | None] = {}
    for name in optional_names:
        values = [getattr(sample, name) for sample in samples]
        optional[name] = (
            None if any(value is None for value in values) else torch.cat(values, dim=0)
        )
    return MeshBatch(
        nodes=torch.cat([sample.nodes for sample in samples], dim=0),
        elements=torch.cat([sample.elements for sample in samples], dim=0),
        features=torch.cat([sample.features for sample in samples], dim=0),
        forces=torch.cat([sample.forces for sample in samples], dim=0),
        constraints=torch.cat([sample.constraints for sample in samples], dim=0),
        stress=optional["stress"],
        displacement=optional["displacement"],
        strain=optional["strain"],
        sample_ids=[identity for sample in samples for identity in sample.sample_ids],
    )
