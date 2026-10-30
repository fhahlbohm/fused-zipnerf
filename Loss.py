"""ZipNeRF/Loss.py"""

import torch

from Optim.Losses.Base import BaseLoss


class ZipNeRFLoss(BaseLoss):
    """Logs the losses computed by the fused training kernels."""

    def __init__(self) -> None:
        super().__init__()
        for name in ('Charbonnier_Color', 'Interlevel', 'Distortion', 'Grid_Regularization'):
            self.add_loss_metric(name, lambda value: value, 1.0)
        self.add_quality_metric('PSNR', lambda mse: -10.0 * torch.log10(mse))

    def forward(self, loss_sums: torch.Tensor, regularization: torch.Tensor) -> torch.Tensor:
        """Logs the loss sums (data, mse, interlevel, distortion) and the grid regularization."""
        return super().forward({
            'Charbonnier_Color': {'value': loss_sums[0]},
            'Interlevel': {'value': loss_sums[2]},
            'Distortion': {'value': loss_sums[3]},
            'Grid_Regularization': {'value': regularization},
            'PSNR': {'mse': loss_sums[1]},
        })
