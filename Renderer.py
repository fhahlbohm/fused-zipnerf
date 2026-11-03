"""ZipNeRF/Renderer.py"""

from typing import Any

import torch

import Framework
from Datasets.utils import View
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.ZipNeRF.Model import ZipNeRFModel
from Methods.ZipNeRF.utils import pixel_radius
from Methods.ZipNeRF.ZipNeRFCudaBackend import Featurizer, NerfMlp, view_rays, view_rays_cached, initialize_rays, \
    composite_and_resample, resample_inference, composite_final, proposal_grid_points, sample_proposal_grid
from Visual.utils import apply_color_map


@Framework.Configurable.configure(
    RAY_BATCH_SIZE=65_536,
    RENDER_EXTRAS=False,  # if True, inference also outputs the opacity and the depth
    USE_MEDIAN_DEPTH=False,  # if True, the depth is the median depth instead of the expected depth
    USE_FAST_INFERENCE=True,  # exact inference kernels; training uses the original sampling and MLP paths
    USE_CACHED_PROPOSALS=False,  # approximate proposal density grids; the final NeRF field is unchanged
    PROPOSAL_GRID_RESOLUTION=1024,  # power of two, at most 1024; both FP32 mip pyramids need about 9.14 GiB at 1024
    PROPOSAL_OPACITY_THRESHOLD=0.995,  # cached rays below this opacity are rendered again with native proposals
)
class ZipNeRFRenderer(BaseRenderer):
    """Renders ZipNeRF with runtime-compiled featurization kernels and bf16 cuBLASLt GEMMs for the NeRF MLP."""

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model, [ZipNeRFModel])
        self.pixel_order_key: tuple[int, int] | None = None
        self.pixel_order_indices: torch.Tensor | None = None
        self.packed_weights: torch.Tensor | None = None
        self.packed_iteration = -1
        self.initial_intervals: dict[tuple[int, int, int, tuple[float, ...]], tuple[torch.Tensor, torch.Tensor]] = {}
        self.proposal_grid_key: tuple | None = None
        self.proposal_pyramids: list[list[torch.Tensor]] = []
        self.proposal_grid_ready: torch.cuda.Event | None = None
        self.proposal_grid_streams: set[int] = set()
        self.last_recovered_rays = 0
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
        proposal_grids: list[list[torch.Tensor]] | None = None,
    ) -> None:
        """Renders a ray batch for inference, writing to the rays' pixels of the image buffers (empty ones are skipped)."""
        warp = self.model.ray_warp(near, far)
        empty = torch.empty(0)
        origins, directions, radii, encoded_directions, sdist, tdist = rays
        n_rays = origins.shape[0]
        n_samples = self.model.N_SAMPLES
        for level, featurizer in enumerate(self.featurizers[:-1]):
            raw_density = torch.empty(n_rays, n_samples[level])
            if proposal_grids is None:
                featurizer.forward(
                    origins, directions, radii, tdist, -1, True, self.model.grids[level].table, scale_magnitudes[level], *self.model.proposal_mlp(level),
                    raw_density, empty, empty, empty, empty)
            else:
                sample_proposal_grid(proposal_grids[level], origins, directions, radii, tdist, raw_density)
            if self.USE_FAST_INFERENCE:
                sdist, tdist = resample_inference(raw_density, sdist, tdist, directions, n_samples[level + 1], warp)
            else:
                _, sdist, tdist = composite_and_resample(raw_density, sdist, tdist, directions, 1.0, n_samples[level + 1], -1, level + 1, warp)
        n_final = n_rays * n_samples[-1]
        features = torch.empty(n_final, self.nerf_mlp.n_features, dtype=torch.bfloat16)
        inputs = empty if self.USE_FAST_INFERENCE else torch.empty(n_final, self.nerf_mlp.n_inputs, dtype=torch.bfloat16)
        self.featurizers[-1].forward(
            origins, directions, radii, tdist, -1, True, self.model.grids[-1].table, scale_magnitudes[-1], empty, empty, empty, empty,
            empty, empty, features, empty if self.USE_FAST_INFERENCE else encoded_directions, inputs)
        if self.USE_FAST_INFERENCE:
            outputs = self.nerf_mlp.forward_rays(features, encoded_directions, nerf_weights, n_samples[-1])
        else:
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

    def clear_proposal_cache(self) -> None:
        """Releases cached proposal grids; queued readers retain their allocations until completion."""
        self.proposal_grid_key = None
        self.proposal_pyramids = []
        self.proposal_grid_ready = None
        self.proposal_grid_streams.clear()

    @torch.no_grad()
    def proposal_grids(self) -> list[list[torch.Tensor]]:
        """Bakes proposal densities lazily, invalidating them after parameter or resolution changes."""
        resolution = self.PROPOSAL_GRID_RESOLUTION
        if not isinstance(resolution, int) or resolution < 2 or resolution > 1024 or resolution & (resolution - 1):
            raise ValueError('PROPOSAL_GRID_RESOLUTION must be a power of two between 2 and 1024')
        reference = self.model.proposal_mlp_params
        parameters = [reference, self.model.scale_magnitudes, *[grid.table for grid in self.model.grids[:-1]]]
        key = (resolution, self.model.num_iterations_trained, reference.device,
            tuple((tensor.data_ptr(), tensor._version) for tensor in parameters))
        stream = torch.cuda.current_stream(reference.device)
        if key != self.proposal_grid_key:
            self.clear_proposal_cache()
            with torch.cuda.device(reference.device):
                count = resolution ** 3
                tables = [torch.empty(count, device=reference.device) for _ in self.featurizers[:-1]]
                pyramids = []
                scales = self.model.level_scale_magnitudes()
                empty = torch.empty(0, device=reference.device)
                batch_size = 262_144
                for start in range(0, count, batch_size):
                    size = min(batch_size, count - start)
                    origins = proposal_grid_points(reference, resolution, start, size)
                    directions = torch.zeros_like(origins)
                    directions[:, 2] = 1
                    radii = torch.zeros(size, device=reference.device)
                    tdist = torch.zeros(size, 2, device=reference.device)
                    raw = torch.empty(size, 1, device=reference.device)
                    for level, featurizer in enumerate(self.featurizers[:-1]):
                        featurizer.forward(origins, directions, radii, tdist, -1, True,
                            self.model.grids[level].table, scales[level], *self.model.proposal_mlp(level),
                            raw, empty, empty, empty, empty)
                        tables[level][start:start + size] = torch.exp((raw[:, 0] - 1).clamp(max=88.72283))
                for table in tables:
                    if not torch.isfinite(table).all().item():
                        raise RuntimeError('Nonfinite proposal density while baking the cache')
                    levels = [table.reshape(resolution, resolution, resolution)]
                    while levels[-1].shape[0] > 1:
                        levels.append(torch.nn.functional.avg_pool3d(levels[-1][None, None], 2, 2)[0, 0].contiguous())
                    if not torch.isfinite(levels[-1]).all().item():
                        raise RuntimeError('Nonfinite proposal density mip pyramid')
                    pyramids.append(levels)
                self.proposal_pyramids = pyramids
                self.proposal_grid_ready = torch.cuda.Event()
                self.proposal_grid_ready.record(stream)
                self.proposal_grid_key = key
        stream.wait_event(self.proposal_grid_ready)
        if stream.cuda_stream not in self.proposal_grid_streams:
            for levels in self.proposal_pyramids:
                for level in levels:
                    level.record_stream(stream)
            self.proposal_grid_streams.add(stream.cuda_stream)
        return self.proposal_pyramids

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
        self.last_recovered_rays = 0
        grids = None
        if self.USE_CACHED_PROPOSALS:
            if not 0.0 <= self.PROPOSAL_OPACITY_THRESHOLD <= 1.0:
                raise ValueError('PROPOSAL_OPACITY_THRESHOLD must be between zero and one')
            grids = self.proposal_grids()
        elif self.proposal_grid_key is not None:
            self.clear_proposal_cache()
        if self.USE_FAST_INFERENCE:
            # keep initialization and reuse on the same device and stream
            key = (local_directions.device.index, torch.cuda.current_stream(local_directions.device).cuda_stream, self.model.N_SAMPLES[0], warp)
            if key not in self.initial_intervals:
                direction = torch.tensor([[0.0, 0.0, 1.0]], device=local_directions.device)
                _, sdist, tdist = initialize_rays(-1, direction, self.model.DIRECTION_ENCODING_DEGREE, self.model.N_SAMPLES[0], warp)
                self.initial_intervals[key] = (sdist, tdist)
        n_pixels = camera.width * camera.height
        empty = torch.empty(0)
        outputs = {'rgb': torch.empty(n_pixels, 3)}
        if self.RENDER_EXTRAS or grids is not None:
            outputs['alpha'] = torch.empty(n_pixels, 1)
        if self.RENDER_EXTRAS:
            outputs['depth'] = torch.empty(n_pixels, 1)

        def render_pixels(pixels: torch.Tensor, proposal_grids: list[list[torch.Tensor]] | None) -> None:
            for start in range(0, pixels.numel(), self.RAY_BATCH_SIZE):
                pixel_indices = pixels[start:start + self.RAY_BATCH_SIZE]
                if self.USE_FAST_INFERENCE:
                    rays = view_rays_cached(pixel_indices, local_directions, c2w, radius, self.model.DIRECTION_ENCODING_DEGREE, *self.initial_intervals[key])
                else:
                    rays = view_rays(pixel_indices, local_directions, c2w, radius, self.model.DIRECTION_ENCODING_DEGREE, self.model.N_SAMPLES[0], warp)
                self.render_rays_inference(
                    rays, camera.near_plane, camera.far_plane, background, scale_magnitudes, nerf_weights, pixel_indices,
                    outputs['rgb'], outputs.get('alpha', empty), outputs.get('depth', empty), proposal_grids)

        render_pixels(order, grids)
        if grids is not None:
            opacity = outputs['alpha'].flatten()
            mask = (opacity < self.PROPOSAL_OPACITY_THRESHOLD) | ~torch.isfinite(opacity) | ~torch.isfinite(outputs['rgb']).all(dim=-1)
            # keep neighboring pixel pairs together for the 64-row fused MLP path
            paired = n_pixels // 2 * 2
            pairs = mask[:paired].reshape(-1, 2).any(dim=-1)
            mask = torch.cat((pairs[:, None].expand(-1, 2).reshape(-1), mask[paired:]))
            pixels = order[mask[order]]
            self.last_recovered_rays = pixels.numel()
            render_pixels(pixels, None)
            if not self.RENDER_EXTRAS:
                del outputs['alpha']
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
