"""Flow matching loss functions."""


import torch
import torch.nn.functional as F

from .base import AbstractLoss


class FlowMatchingLoss(AbstractLoss):
    """Loss for flow matching (e.g., π0 style)."""

    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "mse",
        sigma: float = 0.0,
    ):
        """
        Args:
            weight: Loss weight
            loss_type: Base loss type ("mse", "l1")
            sigma: Noise level for stochastic interpolant
        """
        super().__init__(weight=weight)
        self.loss_type = loss_type
        self.sigma = sigma

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute flow matching loss."""
        # Predicted velocity
        v_pred = predictions.get('velocity', predictions.get('v_pred'))
        # Target velocity (x1 - x0 for optimal transport)
        v_target = targets.get('velocity', targets.get('v_target'))

        if v_pred is None or v_target is None:
            return torch.tensor(0.0)

        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_target)
        else:
            loss = F.l1_loss(v_pred, v_target)

        return loss


class RectifiedFlowLoss(AbstractLoss):
    """Rectified flow loss for straight-line trajectories."""

    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__(weight=weight)
        self.loss_type = loss_type

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        v_pred = predictions.get('velocity')
        x0 = targets.get('x0')  # Noise
        x1 = targets.get('x1')  # Target

        if v_pred is None or x0 is None or x1 is None:
            return torch.tensor(0.0)

        # Target velocity is x1 - x0 (straight line)
        v_target = x1 - x0

        if self.loss_type == "mse":
            return F.mse_loss(v_pred, v_target)
        return F.l1_loss(v_pred, v_target)


class OTCFMLoss(AbstractLoss):
    """Optimal Transport Conditional Flow Matching loss."""

    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 0.0,
    ):
        """
        Args:
            weight: Loss weight
            sigma: Gaussian width for interpolation
        """
        super().__init__(weight=weight)
        self.sigma = sigma

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        v_pred = predictions.get('velocity')
        x0 = targets.get('x0')
        x1 = targets.get('x1')
        targets.get('t')

        if v_pred is None or x0 is None or x1 is None:
            return torch.tensor(0.0)

        # OT-CFM target velocity
        # u_t(x|x0, x1) = x1 - x0
        v_target = x1 - x0

        return F.mse_loss(v_pred, v_target)


class ConsistencyLoss(AbstractLoss):
    """Consistency loss for consistency models."""

    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__(weight=weight)
        self.loss_type = loss_type

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        # f(x_t, t) should equal f(x_s, s) for any s, t on the same trajectory
        f_t = predictions.get('consistency_output')
        f_s = targets.get('consistency_target')

        if f_t is None or f_s is None:
            return torch.tensor(0.0)

        if self.loss_type == "mse":
            return F.mse_loss(f_t, f_s)
        return F.l1_loss(f_t, f_s)


__all__ = [
    'FlowMatchingLoss',
    'RectifiedFlowLoss',
    'OTCFMLoss',
    'ConsistencyLoss',
]
