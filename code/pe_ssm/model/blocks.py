import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PositiveDefiniteProjection(nn.Module):
    def __init__(self, input_dimension: int, matrix_dimension: int) -> None:
        super().__init__()
        self.matrix_dimension = matrix_dimension
        packed = matrix_dimension * (matrix_dimension + 1) // 2
        self.projection = nn.Linear(input_dimension, packed)

    def forward(self, inputs: Tensor) -> Tensor:
        packed = self.projection(inputs)
        shape = (*packed.shape[:-1], self.matrix_dimension, self.matrix_dimension)
        lower = torch.zeros(shape, dtype=packed.dtype, device=packed.device)
        indices = torch.tril_indices(
            self.matrix_dimension, self.matrix_dimension, device=packed.device
        )
        lower[..., indices[0], indices[1]] = packed
        diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
        diagonal.copy_(F.softplus(diagonal) + 1e-5)
        return lower @ lower.transpose(-1, -2)


class StiffnessModulator(nn.Module):
    def __init__(self, state_dimension: int, geometry_dimension: int) -> None:
        super().__init__()
        joint = state_dimension + geometry_dimension
        self.pre = nn.Sequential(
            nn.Linear(joint, state_dimension),
            nn.SiLU(),
            nn.Linear(state_dimension, state_dimension),
            nn.SiLU(),
        )
        self.positive = PositiveDefiniteProjection(state_dimension, 6)
        self.scale = nn.Parameter(torch.tensor(-5.0))

    def forward(self, state: Tensor, geometry: Tensor, base: Tensor) -> Tensor:
        hidden = self.pre(torch.cat((state, geometry), dim=-1))
        delta = self.positive(hidden)
        return base + torch.sigmoid(self.scale) * delta


class SelectivePhysicsScan(nn.Module):
    def __init__(self, state_dimension: int, input_dimension: int) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.input_projection = nn.Linear(input_dimension, state_dimension)
        self.step_projection = nn.Linear(state_dimension, state_dimension)
        self.decay_projection = nn.Linear(state_dimension, state_dimension)
        self.gate_projection = nn.Linear(state_dimension, state_dimension)
        self.output_projection = nn.Linear(state_dimension, state_dimension)
        self.normalization = nn.LayerNorm(state_dimension)

    def forward(
        self, inputs: Tensor, stiffness_summary: Tensor, initial: Tensor | None = None
    ) -> Tensor:
        batch, length, _ = inputs.shape
        state = inputs.new_zeros(batch, self.state_dimension) if initial is None else initial
        projected = self.input_projection(inputs)
        outputs = []
        for index in range(length):
            incoming = projected[:, index]
            step = F.softplus(self.step_projection(state))
            decay = F.softplus(self.decay_projection(stiffness_summary[:, index]))
            transition = torch.exp(-step * decay)
            gate = torch.sigmoid(self.gate_projection(incoming))
            candidate = transition * state + (1.0 - transition) * incoming
            state = gate * candidate + (1.0 - gate) * state
            outputs.append(state)
        stacked = torch.stack(outputs, dim=1)
        return self.output_projection(self.normalization(stacked))


class MultiOrderFusion(nn.Module):
    def __init__(self, state_dimension: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(state_dimension, state_dimension // 2),
            nn.Tanh(),
            nn.Linear(state_dimension // 2, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        weights = torch.softmax(self.score(states), dim=1)
        return torch.sum(weights * states, dim=1)


class CoarseFineTransfer(nn.Module):
    def __init__(self, state_dimension: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(state_dimension * 2, state_dimension),
            nn.GELU(),
            nn.Linear(state_dimension, state_dimension),
        )
        self.norm = nn.LayerNorm(state_dimension)

    def forward(self, coarse: Tensor, fine: Tensor, assignment: Tensor) -> Tensor:
        expanded = coarse.gather(1, assignment[..., None].expand(-1, -1, coarse.shape[-1]))
        return self.norm(fine + self.refine(torch.cat((expanded, fine), dim=-1)))


class ResidualFeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.norm = nn.LayerNorm(dimension)
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.network(self.norm(inputs))
