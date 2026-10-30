"""ZipNeRF/Trainer.py"""

import torch

import Framework
from Logging import Logger
from Datasets.Base import BaseDataset
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback
from Methods.ZipNeRF.Loss import ZipNeRFLoss
from Methods.ZipNeRF.utils import pixel_radius, schlick_bias
from Methods.ZipNeRF.ZipNeRFCudaBackend import Adam, generate_rays, train_final, nerf_mlp_grad_scale
from Optim.lr_utils import LRDecayPolicy


@Framework.Configurable.configure(
    NUM_ITERATIONS=25_000,
    BATCH_SIZE=65_536,
    ANNEAL_SLOPE=10.0,  # slope of the annealing of the proposal weights used for resampling
    LEARNING_RATE=Framework.ConfigParameterList(
        INIT=0.01,
        FINAL=0.001,
        DELAY_STEPS=2_500,
        DELAY_MULT=1e-8,
    ),
    LOSS=Framework.ConfigParameterList(
        CHARBONNIER_PADDING=0.001,
        LAMBDA_INTERLEVEL=0.01,
        INTERLEVEL_BLURS=[0.03, 0.003],
        LAMBDA_DISTORTION=0.01,
        DISTORTION_POWER=-0.25,
        DISTORTION_PREMULT=10_000.0,
        LAMBDA_GRID_REGULARIZATION=0.1,
    ),
)
class ZipNeRFTrainer(GuiTrainer):
    """Defines the trainer for the ZipNeRF method."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        grids = self.model.grids
        buffers = [grid.table for grid in grids] + [self.model.proposal_mlp_params, self.model.nerf_mlp_params]
        for buffer in buffers:
            buffer.grad = torch.zeros_like(buffer)
        self.optimizer = Adam(
            buffers, [buffer.grad for buffer in buffers], [1.0] * (len(grids) + 1) + [nerf_mlp_grad_scale],
            [[offset * grid.n_features_per_level for offset in grid.offsets] for grid in grids] + [[], []],
            self.LOSS.LAMBDA_GRID_REGULARIZATION, grids[0].max_value, 0.9, 0.99, 1e-15, self.model.scale_magnitudes)
        self.lr_policy = LRDecayPolicy(
            lr_init=self.LEARNING_RATE.INIT, lr_final=self.LEARNING_RATE.FINAL, lr_delay_steps=self.LEARNING_RATE.DELAY_STEPS,
            lr_delay_mult=self.LEARNING_RATE.DELAY_MULT, max_steps=self.NUM_ITERATIONS,
        )
        # random numbers of iteration i are hashes of seed_offset + i (seeds must stay below 2^56, see random.cuh)
        self.seed_offset = Framework.config.GLOBAL.RANDOM_SEED << 24
        self.scale_magnitudes = self.model.level_scale_magnitudes()
        self.loss = ZipNeRFLoss()
        self.ray_pool = None

    @pre_training_callback(priority=1000)
    @torch.no_grad()
    def setup_ray_pool(self, _, dataset: BaseDataset) -> None:
        """Gathers the pixels of all training views, from which the rays of each batch are generated."""
        views = list(dataset.train())
        cameras = []
        camera_indices = []
        for view in views:
            if not any(view.camera is camera for camera in cameras):
                cameras.append(view.camera)
            camera_indices.append(next(i for i, camera in enumerate(cameras) if camera is view.camera))
        local_directions = [camera.compute_local_ray_directions() for camera in cameras]
        # argument order of generate_rays
        self.ray_pool = (
            torch.tensor([0] + [view.camera.width * view.camera.height for view in views]).cumsum(0),
            torch.tensor(camera_indices, dtype=torch.int32),
            torch.tensor([0] + [d.shape[0] for d in local_directions]).cumsum(0),
            torch.cat(local_directions).contiguous(),
            torch.stack([view.c2w[:3] for view in views]).contiguous(),
            torch.tensor([pixel_radius(camera) for camera in cameras]),
            torch.cat([(view.rgb.permute(1, 2, 0).reshape(-1, 3) * 255).round().to(torch.uint8) for view in views]),
        )
        Logger.log_info(f'ray pool: {self.ray_pool[-1].shape[0]:,} pixels from {len(views)} views')

    def compute_gradients(
        self,
        rays: tuple[torch.Tensor, ...],
        rgb: torch.Tensor,
        seed: int,
        near: float,
        far: float,
        anneal: float,
        compute_loss_sums: bool,
    ) -> torch.Tensor | None:
        """Accumulates the gradients of a ray batch and returns the loss sums (data, mse, interlevel, distortion) if requested."""
        # the grid regularization is added by the optimizer
        origins, directions, radii = rays[:3]
        state = self.renderer.render_rays_training(*rays, near, far, self.scale_magnitudes, seed, anneal)
        proposals = state['proposals']
        nerf_outputs_grad, proposal_raw_density_grads, loss_sums = train_final(
            state['nerf_outputs'], state['sdist'], state['tdist'],
            [p['raw_density'] for p in proposals], [p['sdist'] for p in proposals], [p['tdist'] for p in proposals], [p['weights'] for p in proposals],
            directions, seed, rgb, self.LOSS.CHARBONNIER_PADDING,
            [self.LOSS.LAMBDA_INTERLEVEL] * len(proposals), self.LOSS.INTERLEVEL_BLURS, self.LOSS.LAMBDA_DISTORTION,
            self.LOSS.DISTORTION_POWER, self.LOSS.DISTORTION_PREMULT, compute_loss_sums)
        # backward of the NeRF level
        features_grad = self.renderer.nerf_mlp.backward(
            nerf_outputs_grad, state['features'], state['nerf_weights'], state['nerf_context'], self.model.nerf_mlp_params, self.model.nerf_mlp_params.grad)
        self.renderer.featurizers[-1].backward(
            origins, directions, radii, state['tdist'], seed, features_grad, self.model.grids[-1].table.grad)
        # backward of the proposal levels
        for level, proposal in enumerate(proposals):
            self.renderer.featurizers[level].backward_mlp(
                origins, directions, radii, proposal['tdist'], seed, proposal['features'], proposal_raw_density_grads[level],
                *self.model.proposal_mlp(level)[:3], self.model.grids[level].table.grad,
                *(self.model.proposal_parameter(level, name, self.model.proposal_mlp_params.grad) for name in ('w0', 'b0', 'w1', 'b1')))
        return loss_sums

    @training_callback(priority=50)
    def training_iteration(self, iteration: int, dataset: BaseDataset) -> None:
        """Performs a training step."""
        self.model.train()
        self.loss.train()
        camera = dataset.default_camera
        seed = self.seed_offset + iteration
        origins, directions, radii, rgb, encoded_directions, sdist, tdist = generate_rays(
            seed, self.BATCH_SIZE, *self.ray_pool, self.model.DIRECTION_ENCODING_DEGREE, self.model.N_SAMPLES[0],
            self.model.ray_warp(camera.near_plane, camera.far_plane))
        loss_sums = self.compute_gradients(
            (origins, directions, radii, encoded_directions, sdist, tdist), rgb, seed, camera.near_plane, camera.far_plane,
            schlick_bias(iteration / self.NUM_ITERATIONS, self.ANNEAL_SLOPE), self.WANDB.ACTIVATE)
        self.optimizer.step(self.lr_policy(iteration))
        if self.WANDB.ACTIVATE:
            self.loss(loss_sums, self.optimizer.regularization())
