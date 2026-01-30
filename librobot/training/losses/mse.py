"""MSE and regression loss functions."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractLoss


class MSELoss(AbstractLoss):
    """Mean Squared Error loss for continuous action prediction."""

    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        mask_key: Optional[str] = None,
    ):
        """
        Args:
            weight: Loss weight
            reduction: Reduction mode ("mean", "sum", "none")
            mask_key: Optional key for loss masking
        """
        super().__init__(weight=weight)
        self.reduction = reduction
        self.mask_key = mask_key
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Compute MSE loss."""
        pred = predictions.get("actions", predictions.get("pred"))
        target = targets.get("actions", targets.get("target"))

        if pred is None or target is None:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)

        loss = self.mse(pred, target)

        # Apply mask if provided
        if self.mask_key and self.mask_key in targets:
            mask = targets[self.mask_key].float()
            loss = loss * mask.unsqueeze(-1)
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * pred.shape[-1] + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class L1Loss(AbstractLoss):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight=weight)
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        pred = predictions.get("actions", predictions.get("pred"))
        target = targets.get("actions", targets.get("target"))

        if pred is None or target is None:
            return torch.tensor(0.0)

        return self.l1(pred, target)


class SmoothL1Loss(AbstractLoss):
    """Smooth L1 (Huber) loss."""

    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        beta: float = 1.0,
    ):
        super().__init__(weight=weight)
        self.reduction = reduction
        self.beta = beta

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        pred = predictions.get("actions", predictions.get("pred"))
        target = targets.get("actions", targets.get("target"))

        if pred is None or target is None:
            return torch.tensor(0.0)

        return F.smooth_l1_loss(pred, target, reduction=self.reduction, beta=self.beta)


class ActionLoss(AbstractLoss):
    """Combined action loss with position and rotation components."""

    def __init__(
        self,
        weight: float = 1.0,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        gripper_weight: float = 1.0,
        position_dims: int = 3,
        rotation_dims: int = 3,
        loss_type: str = "mse",
    ):
        """
        Args:
            weight: Overall weight
            position_weight: Weight for position loss
            rotation_weight: Weight for rotation loss
            gripper_weight: Weight for gripper loss
            position_dims: Number of position dimensions
            rotation_dims: Number of rotation dimensions
            loss_type: Type of loss ("mse", "l1", "smooth_l1")
        """
        super().__init__(weight=weight)
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.gripper_weight = gripper_weight
        self.position_dims = position_dims
        self.rotation_dims = rotation_dims

        if loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        else:
            self.loss_fn = F.smooth_l1_loss

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        pred = predictions.get("actions")
        target = targets.get("actions")

        if pred is None or target is None:
            return torch.tensor(0.0)

        # Split into components
        pos_end = self.position_dims
        rot_end = pos_end + self.rotation_dims

        pos_pred = pred[..., :pos_end]
        pos_target = target[..., :pos_end]

        rot_pred = pred[..., pos_end:rot_end]
        rot_target = target[..., pos_end:rot_end]

        # Gripper (remaining dimensions)
        grip_pred = pred[..., rot_end:]
        grip_target = target[..., rot_end:]

        loss = 0.0

        if self.position_weight > 0:
            loss = loss + self.position_weight * self.loss_fn(pos_pred, pos_target)

        if self.rotation_weight > 0 and rot_pred.shape[-1] > 0:
            loss = loss + self.rotation_weight * self.loss_fn(rot_pred, rot_target)

        if self.gripper_weight > 0 and grip_pred.shape[-1] > 0:
            loss = loss + self.gripper_weight * self.loss_fn(grip_pred, grip_target)

        return loss


__all__ = [
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "ActionLoss",
]
