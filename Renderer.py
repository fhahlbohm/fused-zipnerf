"""ZipNeRF/Renderer.py"""

from typing import Any

import torch

import Framework
from Datasets.utils import View
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.ZipNeRF.Model import ZipNeRFModel
from Methods.ZipNeRF.utils import pixel_radius
from Methods.ZipNeRF.ZipNeRFCudaBackend import Featurizer, NerfMlp, view_rays, composite_and_resample, composite_final
from Visual.utils import apply_color_map


@Framework.Configurable.configure(
    RAY_BATCH_SIZE=65_536,
    RENDER_EXTRAS=False,  # if True, inference also outputs the opacity and the depth
    USE_MEDIAN_DEPTH=False,  # if True, the depth is the median depth instead of the expected depth
)
class ZipNeRFRenderer(BaseRenderer):
    """Renders ZipNeRF with runtime-compiled featurization kernels and bf16 cuBLASLt GEMMs for the NeRF MLP."""

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model, [ZipNeRFModel])
        self.pixel_order_key: tuple[int, int] | None = None
        self.pixel_order_indices: torch.Tensor | None = None
        self.packed_weights: torch.Tensor | None = None
        self.packed_iteration = -1
        # the NeRF level's density MLP is part of the NeRF MLP
        self.featurizers = [
            Featurizer(level, grid.sizes, grid.level_entries, grid.n_features_per_level, self.model.MLP_DENSITY_WIDTH if level < len(self.model.grids) - 1 else 0)
            for level, grid in enumerate(self.model.grids)
        ]
        self.nerf_mlp = NerfMlp(
            self.model.grids[-1].n_output_features, self.model.MLP_DENSITY_WIDTH, self.model.MLP_BOTTLENECK_WIDTH, self.model.MLP_VIEW_WIDTH,
            self.model.DIRECTION_ENCODING_DEGREE)

    def render_rays_training(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        radii: torch.Tensor,
        encoded_directions: torch.Tensor,
        sdist: torch.Tensor,
        tdist: torch.Tensor,
        near: float,
        far: float,
        scale_magnitudes: list[torch.Tensor],
        seed: int,
        anneal: float,
    ) -> dict[str, Any]:
        """Renders a ray batch for training, returning everything the backward needs."""
        warp = self.model.ray_warp(near, far)
        empty = torch.empty(0)
        n_rays = origins.shape[0]
        n_samples = self.model.N_SAMPLES
        rays = (origins, directions, radii)
        proposals = []
        for level, featurizer in enumerate(self.featurizers[:-1]):
            grid = self.model.grids[level]
            proposal = {'sdist': sdist, 'tdist': tdist}
            proposal['raw_density'] = torch.empty(n_rays, n_samples[level])
            proposal['features'] = torch.empty(n_rays * n_samples[level], grid.n_output_features)
            featurizer.forward(
                *rays, tdist, seed, False, grid.table, scale_magnitudes[level], *self.model.proposal_mlp(level),
                proposal['raw_density'], proposal['features'], empty, empty, empty)
            proposal['weights'], sdist, tdist = composite_and_resample(
                proposal['raw_density'], proposal['sdist'], proposal['tdist'], directions, anneal, n_samples[level + 1], seed, level + 1, warp)
            proposals.append(proposal)
        n_final = n_rays * n_samples[-1]
        features = torch.empty(n_final, self.nerf_mlp.n_features, dtype=torch.bfloat16)
        inputs = torch.empty(n_final, self.nerf_mlp.n_inputs, dtype=torch.bfloat16)
        self.featurizers[-1].forward(
            *rays, tdist, seed, False, self.model.grids[-1].table, scale_magnitudes[-1], empty, empty, empty, empty,
            empty, empty, features, encoded_directions, inputs)
        nerf_weights = self.nerf_mlp.pack(self.model.nerf_mlp_params)
        outputs, context = self.nerf_mlp.forward(features, inputs, nerf_weights, True)
        return {
            'proposals': proposals, 'sdist': sdist, 'tdist': tdist, 'features': features, 'nerf_weights': nerf_weights,
            'nerf_outputs': outputs, 'nerf_context': context,
        }

    @torch.no_grad()
    def render_rays_inference(
        self,
        rays: tuple[torch.Tensor, ...],
        near: float,
        far: float,
        background: torch.Tensor,
        scale_magnitudes: list[torch.Tensor],
        nerf_weights: torch.Tensor,
        pixel_indices: torch.Tensor,
        rgb: torch.Tensor,
        alpha: torch.Tensor,
        depth: torch.Tensor,
    ) -> None:
        """Renders a ray batch for inference, writing to the rays' pixels of the image buffers (empty ones are skipped)."""
        warp = self.model.ray_warp(near, far)
        empty = torch.empty(0)
        origins, directions, radii, encoded_directions, sdist, tdist = rays
        n_rays = origins.shape[0]
        n_samples = self.model.N_SAMPLES
        for level, featurizer in enumerate(self.featurizers[:-1]):
            raw_density = torch.empty(n_rays, n_samples[level])
            featurizer.forward(
                origins, directions, radii, tdist, -1, True, self.model.grids[level].table, scale_magnitudes[level], *self.model.proposal_mlp(level),
                raw_density, empty, empty, empty, empty)
            _, sdist, tdist = composite_and_resample(raw_density, sdist, tdist, directions, 1.0, n_samples[level + 1], -1, level + 1, warp)
        n_final = n_rays * n_samples[-1]
        features = torch.empty(n_final, self.nerf_mlp.n_features, dtype=torch.bfloat16)
        inputs = torch.empty(n_final, self.nerf_mlp.n_inputs, dtype=torch.bfloat16)
        self.featurizers[-1].forward(
            origins, directions, radii, tdist, -1, True, self.model.grids[-1].table, scale_magnitudes[-1], empty, empty, empty, empty,
            empty, empty, features, encoded_directions, inputs)
        outputs, _ = self.nerf_mlp.forward(features, inputs, nerf_weights, False)
        composite_final(outputs, tdist, directions, background, pixel_indices, rgb, alpha, depth, self.USE_MEDIAN_DEPTH)

    def pixel_order(self, width: int, height: int) -> torch.Tensor:
        """Returns the pixel indices in Morton order, so that each ray batch covers compact image tiles."""
        if self.pixel_order_key != (width, height):
            def spread(v: torch.Tensor) -> torch.Tensor:
                for shift, mask in ((8, 0x00FF00FF), (4, 0x0F0F0F0F), (2, 0x33333333), (1, 0x55555555)):
                    v = (v | (v << shift)) & mask
                return v
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            self.pixel_order_indices = (spread(x) | (spread(y) << 1)).flatten().argsort()
            self.pixel_order_key = (width, height)
        return self.pixel_order_indices

    def nerf_weights(self) -> torch.Tensor:
        """Returns the packed NeRF MLP weights, repacking them only after training iterations."""
        if self.packed_iteration != self.model.num_iterations_trained:
            self.packed_weights = self.nerf_mlp.pack(self.model.nerf_mlp_params)
            self.packed_iteration = self.model.num_iterations_trained
        return self.packed_weights

    @torch.no_grad()
    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        camera = view.camera
        order = self.pixel_order(camera.width, camera.height)
        local_directions = camera.compute_local_ray_directions()
        c2w = view.c2w[:3].contiguous()
        radius = pixel_radius(camera)
        warp = self.model.ray_warp(camera.near_plane, camera.far_plane)
        background = camera.background_color.to(local_directions.device)
        scale_magnitudes, nerf_weights = self.model.level_scale_magnitudes(), self.nerf_weights()
        n_pixels = camera.width * camera.height
        empty = torch.empty(0)
        outputs = {'rgb': torch.empty(n_pixels, 3)}
        if self.RENDER_EXTRAS:
            outputs |= {'alpha': torch.empty(n_pixels, 1), 'depth': torch.empty(n_pixels, 1)}
        for start in range(0, n_pixels, self.RAY_BATCH_SIZE):
            pixel_indices = order[start:start + self.RAY_BATCH_SIZE]
            rays = view_rays(pixel_indices, local_directions, c2w, radius, self.model.DIRECTION_ENCODING_DEGREE, self.model.N_SAMPLES[0], warp)
            self.render_rays_inference(
                rays, camera.near_plane, camera.far_plane, background, scale_magnitudes, nerf_weights, pixel_indices,
                outputs['rgb'], outputs.get('alpha', empty), outputs.get('depth', empty))
        for key in outputs:
            outputs[key] = outputs[key].reshape(camera.height, camera.width, -1)
            if to_chw:
                outputs[key] = outputs[key].permute(2, 0, 1)
        return outputs

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        outputs_color = {'rgb': outputs['rgb'].clamp(0.0, 1.0)}
        if 'alpha' in outputs:
            outputs_color |= {
                'alpha': outputs['alpha'].expand_as(outputs['rgb']),
                'depth': apply_color_map(color_map='SPECTRAL', image=outputs['depth'], mask=outputs['alpha'], interpolate=True),
            }
        return outputs_color
