"""Cross-entropy and classification losses."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractLoss


class CrossEntropyLoss(AbstractLoss):
    """Cross-entropy loss for discrete action prediction."""
    
    def __init__(
        self,
        weight: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Args:
            weight: Loss weight
            ignore_index: Index to ignore in loss
            label_smoothing: Label smoothing factor
            reduction: Reduction mode
        """
        super().__init__(weight=weight)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        logits = predictions.get('logits', predictions.get('action_logits'))
        labels = targets.get('labels', targets.get('action_tokens'))
        
        if logits is None or labels is None:
            return torch.tensor(0.0)
        
        # Reshape for cross-entropy if needed
        if logits.dim() == 3:  # [B, T, V]
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
        
        return self.ce(logits, labels)


class FocalLoss(AbstractLoss):
    """Focal loss for handling class imbalance."""
    
    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            weight: Loss weight
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter
            reduction: Reduction mode
        """
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        logits = predictions.get('logits')
        labels = targets.get('labels')
        
        if logits is None or labels is None:
            return torch.tensor(0.0)
        
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (self.alpha * (1 - pt) ** self.gamma)
        
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TokenLoss(AbstractLoss):
    """Loss for autoregressive token prediction (e.g., RT-2 style)."""
    
    def __init__(
        self,
        weight: float = 1.0,
        vocab_size: int = 256,
        num_action_tokens: int = 7,
        ignore_index: int = -100,
    ):
        """
        Args:
            weight: Loss weight
            vocab_size: Vocabulary size
            num_action_tokens: Number of action tokens per step
            ignore_index: Index to ignore
        """
        super().__init__(weight=weight)
        self.vocab_size = vocab_size
        self.num_action_tokens = num_action_tokens
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        logits = predictions.get('action_logits')
        tokens = targets.get('action_tokens')
        
        if logits is None or tokens is None:
            return torch.tensor(0.0)
        
        # Ensure correct shapes
        B = logits.shape[0]
        
        if logits.dim() == 3:  # [B, T, V]
            logits = logits.reshape(-1, logits.shape[-1])
        
        if tokens.dim() == 2:  # [B, T]
            tokens = tokens.reshape(-1)
        
        return self.ce(logits, tokens.long())


class BCELoss(AbstractLoss):
    """Binary cross-entropy loss for binary classification."""
    
    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(weight=weight)
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        logits = predictions.get('logits')
        labels = targets.get('labels')
        
        if logits is None or labels is None:
            return torch.tensor(0.0)
        
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(),
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )


__all__ = [
    'CrossEntropyLoss',
    'FocalLoss',
    'TokenLoss',
    'BCELoss',
]
