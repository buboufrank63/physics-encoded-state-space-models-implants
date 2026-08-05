import torch
from torch import Tensor, nn

from pe_ssm.model.blocks import MultiOrderFusion, ResidualFeedForward, SelectivePhysicsScan
from pe_ssm.physics.elasticity import isotropic_stiffness, stress_from_strain
from pe_ssm.schema import ModelConfig, Prediction


class PhysicsLayer(nn.Module):
    def __init__(self, dimension: int, input_dimension: int) -> None:
        super().__init__()
        self.scan = SelectivePhysicsScan(dimension, input_dimension)
        self.feed_forward = ResidualFeedForward(dimension)
        self.stiffness_embed = nn.Sequential(
            nn.Linear(21, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )

    def forward(self, inputs: Tensor, stiffness: Tensor, initial: Tensor | None = None) -> Tensor:
        triangular = torch.tril_indices(6, 6, device=stiffness.device)
        packed = stiffness[..., triangular[0], triangular[1]]
        summary = self.stiffness_embed(packed)
        return self.feed_forward(self.scan(inputs, summary, initial))


class PhysicsEncodedStateSpaceModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        dimension = self.config.state_dimension
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.config.input_dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.layers = nn.ModuleList(
            [PhysicsLayer(dimension, dimension) for _ in range(self.config.layers)]
        )
        self.order_fusion = MultiOrderFusion(dimension)
        self.stress_decoder = nn.Linear(dimension, 6)
        self.displacement_decoder = nn.Linear(dimension, 3)
        self.strain_decoder = nn.Linear(dimension, 6)
        self.young_modulus: Tensor
        self.poisson_ratio: Tensor
        self.register_buffer("young_modulus", torch.tensor(120000.0))
        self.register_buffer("poisson_ratio", torch.tensor(0.34))

    def constitutive(self, features: Tensor) -> Tensor:
        young = self.young_modulus.expand(features.shape[:-1])
        poisson = self.poisson_ratio.expand(features.shape[:-1])
        return isotropic_stiffness(young, poisson)

    def forward(self, features: Tensor, orders: Tensor | None = None) -> Prediction:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, elements, channels]")
        encoded = self.feature_encoder(features)
        stiffness = self.constitutive(features)
        if orders is None:
            orders = torch.arange(features.shape[1], device=features.device)[None]
        order_states = []
        for order in orders:
            selected = order.to(features.device)
            state = encoded.index_select(1, selected)
            selected_stiffness = stiffness.index_select(1, selected)
            for layer in self.layers:
                state = layer(state, selected_stiffness)
            inverse = torch.empty_like(selected)
            inverse.scatter_(0, selected, torch.arange(selected.numel(), device=selected.device))
            order_states.append(state.index_select(1, inverse))
        hidden = self.order_fusion(torch.stack(order_states, dim=1))
        raw_stress = self.stress_decoder(hidden)
        strain = self.strain_decoder(hidden)
        encoded_stress = stress_from_strain(stiffness, strain)
        stress = raw_stress + encoded_stress
        displacement = self.displacement_decoder(hidden)
        diagonal = torch.diagonal(stiffness, dim1=-2, dim2=-1)
        transition = torch.exp(-diagonal / diagonal.amax(dim=-1, keepdim=True).clamp_min(1e-8))
        return Prediction(
            stress=stress,
            strain=strain,
            displacement=displacement,
            hidden=hidden,
            stiffness=stiffness,
            transition=transition,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
