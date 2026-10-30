"""ZipNeRF/Model.py"""

import math

import numpy as np
import torch

import Framework
from Methods.Base.Model import BaseModel
from Methods.ZipNeRF.utils import next_multiple, ray_warp
from Methods.ZipNeRF.ZipNeRFCudaBackend import grid_max_value, nerf_mlp_parameters


class HashGrid(torch.nn.Module):
    """Stores a multiresolution hash grid (dense coarse levels) in a single table."""

    def __init__(self, n_levels: int, n_features_per_level: int, log2_hashmap_size: int, base_resolution: int, per_level_scale: float) -> None:
        super().__init__()
        self.sizes = [round(base_resolution * per_level_scale ** level) for level in range(n_levels)]
        self.n_features_per_level = n_features_per_level
        self.level_entries = [min(size ** 3, 2 ** log2_hashmap_size) for size in self.sizes]
        self.offsets = np.cumsum([0] + self.level_entries).tolist()
        self.max_value = grid_max_value
        self.table = torch.nn.Parameter(torch.empty(self.offsets[-1], n_features_per_level).uniform_(-self.max_value, self.max_value))

    @property
    def n_levels(self) -> int:
        return len(self.sizes)

    @property
    def n_output_features(self) -> int:
        return self.n_levels * (self.n_features_per_level + 1)

    def level_tables(self) -> list[torch.Tensor]:
        return [self.table[start:end] for start, end in zip(self.offsets[:-1], self.offsets[1:])]

    @torch.no_grad()
    def scale_feature_magnitudes(self) -> torch.Tensor:
        """Computes the per-level magnitudes of the appended scale feature."""
        return torch.stack([(self.max_value ** 2 + level.square().mean()).sqrt() for level in self.level_tables()])


class FlatParameterLayout:
    """Provides named tensor views into a flat parameter buffer."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.size = 0

    def add(self, name: str, *shape: int) -> None:
        self.entries[name] = (self.size, shape)
        self.size += math.prod(shape)

    def view(self, buffer: torch.Tensor, name: str) -> torch.Tensor:
        offset, shape = self.entries[name]
        return buffer[offset:offset + math.prod(shape)].view(shape)


def he_uniform_(weight: torch.Tensor, fan_in: int) -> None:
    """Initializes the first fan_in input columns He-uniformly (flax default)."""
    bound = math.sqrt(6 / fan_in)
    weight[:, :fan_in].uniform_(-bound, bound)


@Framework.Configurable.configure(
    N_SAMPLES=[64, 64, 32],  # samples per ray for the two proposal levels and the NeRF level
    RAYDIST_POWER=-1.5,  # power ladder curve that spaces the samples along the rays (must not be 0 or 1)
    RAYDIST_PREMULT=2.0,
    HASHGRID_N_LEVELS=[6, 8, 10],  # resolution levels of the hash grid of each sampling level
    HASHGRID_N_FEATURES_PER_LEVEL=[1, 1, 4],
    HASHGRID_LOG2_HASHMAP_SIZE=21,
    HASHGRID_BASE_RESOLUTION=16,
    HASHGRID_PER_LEVEL_SCALE=2.0,
    MLP_DENSITY_WIDTH=64,  # hidden width of the density MLPs of all levels
    MLP_BOTTLENECK_WIDTH=256,
    MLP_VIEW_WIDTH=256,  # width of the three view-dependent layers of the NeRF MLP
    DIRECTION_ENCODING_DEGREE=4,
)
class ZipNeRFModel(BaseModel):
    """Defines the ZipNeRF model (https://github.com/jonbarron/camp_zipnerf): a hash grid and an MLP per sampling level."""

    def build(self) -> 'ZipNeRFModel':
        """Builds the model."""
        self.grids = torch.nn.ModuleList([
            HashGrid(n_levels, n_features_per_level, self.HASHGRID_LOG2_HASHMAP_SIZE, self.HASHGRID_BASE_RESOLUTION, self.HASHGRID_PER_LEVEL_SCALE)
            for n_levels, n_features_per_level in zip(self.HASHGRID_N_LEVELS, self.HASHGRID_N_FEATURES_PER_LEVEL)
        ])
        # proposal density MLPs: features -> MLP_DENSITY_WIDTH (relu) -> 1
        self.proposal_layout = FlatParameterLayout()
        for level, grid in enumerate(self.grids[:-1]):
            self.proposal_layout.add(f'{level}.w0', self.MLP_DENSITY_WIDTH, grid.n_output_features)
            self.proposal_layout.add(f'{level}.b0', self.MLP_DENSITY_WIDTH)
            self.proposal_layout.add(f'{level}.w1', self.MLP_DENSITY_WIDTH)
            self.proposal_layout.add(f'{level}.b1', 1)
        # NeRF MLP: density layers, bottleneck, and three view layers with a skip connection to the rgb layer
        nerf_parameters = nerf_mlp_parameters(
            self.grids[-1].n_output_features, self.MLP_DENSITY_WIDTH, self.MLP_BOTTLENECK_WIDTH, self.MLP_VIEW_WIDTH, self.DIRECTION_ENCODING_DEGREE)
        self.nerf_layout = FlatParameterLayout()
        for name, shape, _ in nerf_parameters:
            self.nerf_layout.add(name, *shape)
        # magnitudes of the appended scale features of all grid levels (updated by the optimizer)
        self.register_buffer('scale_magnitudes', torch.cat([grid.scale_feature_magnitudes() for grid in self.grids]))
        # flat MLP parameter buffers, padded to multiples of four for the vectorized optimizer
        self.proposal_mlp_params = torch.nn.Parameter(torch.zeros(next_multiple(self.proposal_layout.size, 4)))
        self.nerf_mlp_params = torch.nn.Parameter(torch.zeros(next_multiple(self.nerf_layout.size, 4)))
        with torch.no_grad():
            for level, grid in enumerate(self.grids[:-1]):
                he_uniform_(self.proposal_parameter(level, 'w0'), grid.n_output_features)
                he_uniform_(self.proposal_parameter(level, 'w1')[None], self.MLP_DENSITY_WIDTH)
            for name, _, fan_in in nerf_parameters:
                # zero padding columns come last and never receive gradients
                if fan_in > 0:
                    he_uniform_(self.nerf_parameter(name), fan_in)
        return self

    def ray_warp(self, near: float, far: float) -> tuple[float, ...]:
        """Returns the constants of the mapping between normalized and metric ray distances."""
        return ray_warp(near, far, self.RAYDIST_POWER, self.RAYDIST_PREMULT)

    def level_scale_magnitudes(self) -> list[torch.Tensor]:
        """Returns views of the scale feature magnitudes per grid."""
        return list(self.scale_magnitudes.split([grid.n_levels for grid in self.grids]))

    def proposal_parameter(self, level: int, name: str, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Returns a proposal MLP tensor as a view into the parameter buffer or the given buffer."""
        return self.proposal_layout.view(self.proposal_mlp_params if buffer is None else buffer, f'{level}.{name}')

    def proposal_mlp(self, level: int) -> tuple[torch.Tensor, ...]:
        """Returns the tensors (w0, b0, w1, b1) of a proposal MLP."""
        return tuple(self.proposal_parameter(level, name) for name in ('w0', 'b0', 'w1', 'b1'))

    def nerf_parameter(self, name: str, buffer: torch.Tensor | None = None) -> torch.Tensor:
        """Returns a NeRF MLP tensor as a view into the parameter buffer or the given buffer."""
        return self.nerf_layout.view(self.nerf_mlp_params if buffer is None else buffer, name)
