"""Action buffering and smoothing for stable robot control."""

from collections import deque
from typing import Any, Deque, List, Optional, Union
import numpy as np
import torch
from librobot.utils import get_logger


logger = get_logger(__name__)


class ActionBuffer:
    """
    Buffer for storing and smoothing action predictions.
    
    Maintains a sliding window of actions and applies smoothing
    to reduce jitter and improve control stability.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        smoothing_method: str = "exponential",
        alpha: float = 0.3,
    ):
        """
        Initialize action buffer.
        
        Args:
            buffer_size: Maximum number of actions to store
            smoothing_method: Smoothing method to use:
                - "none": No smoothing
                - "mean": Moving average
                - "median": Moving median
                - "exponential": Exponential moving average
                - "gaussian": Gaussian-weighted average
            alpha: Smoothing factor for exponential smoothing (0 < alpha <= 1)
                Lower alpha = more smoothing
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in range (0, 1]")
        
        self.buffer_size = buffer_size
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        
        self._buffer: Deque[Union[np.ndarray, torch.Tensor]] = deque(maxlen=buffer_size)
        self._smoothed_action: Optional[Union[np.ndarray, torch.Tensor]] = None
    
    def add(self, action: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Add action to buffer.
        
        Args:
            action: Action array/tensor to add
        """
        if not isinstance(action, (np.ndarray, torch.Tensor)):
            raise TypeError("Action must be numpy array or torch tensor")
        
        self._buffer.append(action)
        logger.debug(f"Added action to buffer (size: {len(self._buffer)})")
    
    def get_smoothed(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Get smoothed action from buffer.
        
        Returns:
            Smoothed action, or None if buffer is empty
        """
        if not self._buffer:
            return None
        
        if self.smoothing_method == "none":
            return self._buffer[-1]
        elif self.smoothing_method == "mean":
            return self._smooth_mean()
        elif self.smoothing_method == "median":
            return self._smooth_median()
        elif self.smoothing_method == "exponential":
            return self._smooth_exponential()
        elif self.smoothing_method == "gaussian":
            return self._smooth_gaussian()
        else:
            logger.warning(f"Unknown smoothing method: {self.smoothing_method}, using 'none'")
            return self._buffer[-1]
    
    def add_and_smooth(
        self,
        action: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add action and return smoothed result.
        
        Args:
            action: Action to add
            
        Returns:
            Smoothed action
        """
        self.add(action)
        smoothed = self.get_smoothed()
        if smoothed is None:
            return action
        return smoothed
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._smoothed_action = None
        logger.debug("Action buffer cleared")
    
    def _smooth_mean(self) -> Union[np.ndarray, torch.Tensor]:
        """Apply moving average smoothing."""
        if isinstance(self._buffer[0], torch.Tensor):
            stacked = torch.stack(list(self._buffer), dim=0)
            return stacked.mean(dim=0)
        else:
            stacked = np.stack(list(self._buffer), axis=0)
            return np.mean(stacked, axis=0)
    
    def _smooth_median(self) -> Union[np.ndarray, torch.Tensor]:
        """Apply moving median smoothing."""
        if isinstance(self._buffer[0], torch.Tensor):
            stacked = torch.stack(list(self._buffer), dim=0)
            return stacked.median(dim=0)[0]
        else:
            stacked = np.stack(list(self._buffer), axis=0)
            return np.median(stacked, axis=0)
    
    def _smooth_exponential(self) -> Union[np.ndarray, torch.Tensor]:
        """Apply exponential moving average smoothing."""
        if self._smoothed_action is None:
            self._smoothed_action = self._buffer[0]
        
        current_action = self._buffer[-1]
        
        if isinstance(current_action, torch.Tensor):
            self._smoothed_action = (
                self.alpha * current_action + 
                (1 - self.alpha) * self._smoothed_action
            )
        else:
            self._smoothed_action = (
                self.alpha * current_action + 
                (1 - self.alpha) * self._smoothed_action
            )
        
        return self._smoothed_action
    
    def _smooth_gaussian(self) -> Union[np.ndarray, torch.Tensor]:
        """Apply Gaussian-weighted smoothing."""
        # Generate Gaussian weights
        buffer_len = len(self._buffer)
        sigma = buffer_len / 4.0  # Standard deviation
        
        # Center weights around the most recent action
        x = np.arange(buffer_len)
        weights = np.exp(-0.5 * ((x - (buffer_len - 1)) / sigma) ** 2)
        weights = weights / weights.sum()
        
        if isinstance(self._buffer[0], torch.Tensor):
            stacked = torch.stack(list(self._buffer), dim=0)
            weights_tensor = torch.tensor(
                weights, 
                dtype=stacked.dtype, 
                device=stacked.device
            ).reshape(-1, *([1] * (stacked.ndim - 1)))
            return (stacked * weights_tensor).sum(dim=0)
        else:
            stacked = np.stack(list(self._buffer), axis=0)
            weights = weights.reshape(-1, *([1] * (stacked.ndim - 1)))
            return (stacked * weights).sum(axis=0)
    
    def get_size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of actions in buffer
        """
        return len(self._buffer)
    
    def is_full(self) -> bool:
        """
        Check if buffer is full.
        
        Returns:
            True if buffer is at maximum capacity
        """
        return len(self._buffer) == self.buffer_size
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self._buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ActionBuffer(size={len(self._buffer)}/{self.buffer_size}, "
            f"method={self.smoothing_method})"
        )


class TemporalEnsembleBuffer:
    """
    Buffer for temporal ensemble of action predictions.
    
    Maintains multiple action sequences predicted at different timesteps
    and aggregates them for more stable predictions.
    """
    
    def __init__(
        self,
        action_horizon: int,
        ensemble_size: int = 5,
        aggregation: str = "weighted_mean",
    ):
        """
        Initialize temporal ensemble buffer.
        
        Args:
            action_horizon: Number of future actions predicted
            ensemble_size: Number of predictions to maintain
            aggregation: Aggregation method ("mean", "weighted_mean", "median")
        """
        self.action_horizon = action_horizon
        self.ensemble_size = ensemble_size
        self.aggregation = aggregation
        
        # Buffer stores list of action sequences with their timestamps
        # Each entry: (timestamp, action_sequence)
        self._buffer: Deque = deque(maxlen=ensemble_size)
        self._current_step = 0
    
    def add_prediction(
        self,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Add new action prediction sequence.
        
        Args:
            actions: Action sequence of shape (action_horizon, action_dim)
        """
        self._buffer.append((self._current_step, actions))
        self._current_step += 1
    
    def get_action(self, offset: int = 0) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Get aggregated action for current timestep.
        
        Args:
            offset: Offset from current timestep (0 = current, 1 = next, etc.)
            
        Returns:
            Aggregated action for the specified timestep
        """
        if not self._buffer:
            return None
        
        # Collect actions from all sequences that have prediction for current step
        valid_actions = []
        weights = []
        
        for pred_step, action_seq in self._buffer:
            # Calculate how many steps ahead this prediction is
            steps_ahead = self._current_step - pred_step
            action_idx = steps_ahead + offset
            
            # Check if this prediction covers the requested action
            if 0 <= action_idx < len(action_seq):
                valid_actions.append(action_seq[action_idx])
                
                # Newer predictions get higher weight
                if self.aggregation == "weighted_mean":
                    weight = np.exp(-0.5 * (steps_ahead / self.ensemble_size) ** 2)
                    weights.append(weight)
        
        if not valid_actions:
            return None
        
        # Aggregate actions
        if self.aggregation == "mean":
            return self._aggregate_mean(valid_actions)
        elif self.aggregation == "weighted_mean":
            return self._aggregate_weighted_mean(valid_actions, weights)
        elif self.aggregation == "median":
            return self._aggregate_median(valid_actions)
        else:
            return valid_actions[-1]  # Return most recent
    
    def _aggregate_mean(
        self,
        actions: List[Union[np.ndarray, torch.Tensor]]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Compute mean of actions."""
        if isinstance(actions[0], torch.Tensor):
            return torch.stack(actions).mean(dim=0)
        else:
            return np.stack(actions).mean(axis=0)
    
    def _aggregate_weighted_mean(
        self,
        actions: List[Union[np.ndarray, torch.Tensor]],
        weights: List[float]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Compute weighted mean of actions."""
        weights_normalized = np.array(weights) / np.sum(weights)
        
        if isinstance(actions[0], torch.Tensor):
            stacked = torch.stack(actions)
            weights_tensor = torch.tensor(
                weights_normalized,
                dtype=stacked.dtype,
                device=stacked.device
            ).reshape(-1, *([1] * (stacked.ndim - 1)))
            return (stacked * weights_tensor).sum(dim=0)
        else:
            stacked = np.stack(actions)
            weights_arr = weights_normalized.reshape(-1, *([1] * (stacked.ndim - 1)))
            return (stacked * weights_arr).sum(axis=0)
    
    def _aggregate_median(
        self,
        actions: List[Union[np.ndarray, torch.Tensor]]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Compute median of actions."""
        if isinstance(actions[0], torch.Tensor):
            return torch.stack(actions).median(dim=0)[0]
        else:
            return np.median(np.stack(actions), axis=0)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._current_step = 0
    
    def step(self) -> None:
        """Advance to next timestep."""
        self._current_step += 1


class AdaptiveActionBuffer(ActionBuffer):
    """
    Action buffer with adaptive smoothing based on action variance.
    
    Automatically adjusts smoothing strength based on prediction stability.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        min_alpha: float = 0.1,
        max_alpha: float = 0.9,
        variance_threshold: float = 0.1,
    ):
        """
        Initialize adaptive action buffer.
        
        Args:
            buffer_size: Maximum number of actions to store
            min_alpha: Minimum smoothing factor (more smoothing)
            max_alpha: Maximum smoothing factor (less smoothing)
            variance_threshold: Variance threshold for adaptation
        """
        super().__init__(
            buffer_size=buffer_size,
            smoothing_method="exponential",
            alpha=max_alpha
        )
        
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.variance_threshold = variance_threshold
    
    def get_smoothed(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Get smoothed action with adaptive alpha."""
        if len(self._buffer) < 2:
            return super().get_smoothed()
        
        # Calculate variance of recent actions
        if isinstance(self._buffer[0], torch.Tensor):
            recent = torch.stack(list(self._buffer)[-5:], dim=0)
            variance = recent.var(dim=0).mean().item()
        else:
            recent = np.stack(list(self._buffer)[-5:], axis=0)
            variance = np.var(recent, axis=0).mean()
        
        # Adapt alpha based on variance
        # High variance -> lower alpha (more smoothing)
        # Low variance -> higher alpha (less smoothing)
        if variance > self.variance_threshold:
            self.alpha = self.min_alpha
        else:
            self.alpha = self.max_alpha
        
        logger.debug(f"Adaptive alpha: {self.alpha:.3f} (variance: {variance:.4f})")
        
        return super().get_smoothed()
