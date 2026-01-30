"""Policy wrapper for inference with VLA models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from librobot.utils import get_logger, load_checkpoint


logger = get_logger(__name__)


class BasePolicy(ABC):
    """
    Abstract base class for VLA policies.
    
    Provides a unified interface for loading and running inference
    with Vision-Language-Action models.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize policy.
        
        Args:
            model: PyTorch model for inference
            device: Device to run inference on (cuda/cpu)
            dtype: Data type for inference (float32/float16/bfloat16)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self._is_loaded = False
        
        if self.model is not None:
            self.model = self.model.to(self.device, dtype=self.dtype)
            self.model.eval()
            self._is_loaded = True
    
    @abstractmethod
    def predict(
        self,
        observation: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on observation.
        
        Args:
            observation: Dictionary containing images, text, state etc.
            **kwargs: Additional inference arguments
            
        Returns:
            Dictionary containing predicted actions and metadata
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset policy state (KV cache, action buffer, etc.)."""
        pass
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
        **kwargs
    ) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, strictly enforce state dict keys match
            **kwargs: Additional loading arguments
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Cannot load checkpoint.")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(str(checkpoint_path))
        
        # Load model state
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=strict)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self._is_loaded = True
        
        logger.info("Checkpoint loaded successfully")
    
    @torch.no_grad()
    def __call__(
        self,
        observation: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenient call interface for prediction.
        
        Args:
            observation: Dictionary containing observation data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions
        """
        return self.predict(observation, **kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded
    
    def to(self, device: Union[str, torch.device]) -> "BasePolicy":
        """
        Move policy to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self


class VLAPolicy(BasePolicy):
    """
    Vision-Language-Action policy for robotics.
    
    Supports transformer-based VLA models with optional KV caching,
    action buffering, and quantization.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        use_kv_cache: bool = False,
        action_horizon: int = 1,
        use_action_buffer: bool = False,
        temporal_ensemble: bool = False,
    ):
        """
        Initialize VLA policy.
        
        Args:
            model: VLA model
            device: Device for inference
            dtype: Data type for inference
            use_kv_cache: Enable KV cache for efficient inference
            action_horizon: Number of future actions to predict
            use_action_buffer: Enable action buffering/smoothing
            temporal_ensemble: Use temporal ensemble for predictions
        """
        super().__init__(model=model, device=device, dtype=dtype)
        
        self.use_kv_cache = use_kv_cache
        self.action_horizon = action_horizon
        self.use_action_buffer = use_action_buffer
        self.temporal_ensemble = temporal_ensemble
        
        # Initialize optional components
        self.kv_cache = None
        self.action_buffer = None
        
        if use_kv_cache:
            from .kv_cache import KVCache
            self.kv_cache = KVCache()
        
        if use_action_buffer:
            from .action_buffer import ActionBuffer
            self.action_buffer = ActionBuffer(
                buffer_size=action_horizon,
                smoothing_method="exponential"
            )
    
    @torch.no_grad()
    def predict(
        self,
        observation: Dict[str, Any],
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict actions from observation.
        
        Args:
            observation: Dictionary with keys:
                - image: torch.Tensor or numpy array of shape (H, W, C) or (C, H, W)
                - text: Optional[str] - instruction text
                - state: Optional[torch.Tensor] - robot state
            return_logits: If True, return raw logits
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - actions: Predicted actions (action_horizon, action_dim)
                - logits: Optional raw logits
                - metadata: Additional prediction metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")
        
        # TODO: Implement observation preprocessing
        # - Convert images to tensors
        # - Normalize and resize images
        # - Tokenize text instructions
        # - Prepare state information
        
        # TODO: Implement forward pass through model
        # - Handle KV cache if enabled
        # - Support multi-modal inputs (vision + language)
        # - Generate actions autoregressively if needed
        
        # TODO: Implement action post-processing
        # - Denormalize actions
        # - Apply action buffer/smoothing
        # - Apply temporal ensemble if enabled
        
        # Placeholder implementation
        batch_inputs = self._prepare_inputs(observation)
        
        # Forward pass
        if self.use_kv_cache and self.kv_cache is not None:
            outputs = self.model(
                **batch_inputs,
                past_key_values=self.kv_cache.get(),
                use_cache=True
            )
            self.kv_cache.update(outputs.get("past_key_values"))
        else:
            outputs = self.model(**batch_inputs)
        
        # Extract actions
        actions = self._extract_actions(outputs)
        
        # Apply action buffer
        if self.use_action_buffer and self.action_buffer is not None:
            actions = self.action_buffer.add_and_smooth(actions)
        
        result = {
            "actions": actions,
            "metadata": {
                "action_horizon": self.action_horizon,
                "device": str(self.device),
            }
        }
        
        if return_logits:
            result["logits"] = outputs.get("logits")
        
        return result
    
    def reset(self) -> None:
        """Reset policy state."""
        if self.kv_cache is not None:
            self.kv_cache.clear()
        
        if self.action_buffer is not None:
            self.action_buffer.clear()
        
        logger.debug("Policy state reset")
    
    def _prepare_inputs(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs from observation.
        
        Args:
            observation: Raw observation dictionary
            
        Returns:
            Dictionary of preprocessed tensors ready for model
        """
        # TODO: Implement full input preparation pipeline
        # - Image preprocessing (resize, normalize)
        # - Text tokenization
        # - State tensor preparation
        # - Batching and device placement
        
        inputs = {}
        
        if "image" in observation:
            image = observation["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image)
            
            # Ensure correct shape (B, C, H, W)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
                image = image.permute(0, 3, 1, 2)
            
            inputs["images"] = image.to(self.device, dtype=self.dtype)
        
        if "text" in observation:
            # TODO: Tokenize text instruction
            inputs["text"] = observation["text"]
        
        if "state" in observation:
            state = observation["state"]
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state)
            inputs["state"] = state.to(self.device, dtype=self.dtype)
        
        return inputs
    
    def _extract_actions(self, model_outputs: Any) -> torch.Tensor:
        """
        Extract action predictions from model outputs.
        
        Args:
            model_outputs: Raw model outputs
            
        Returns:
            Action tensor of shape (action_horizon, action_dim)
        """
        # TODO: Implement action extraction logic
        # - Handle different model output formats
        # - Support continuous and discrete actions
        # - Handle action chunking for temporal ensemble
        
        if isinstance(model_outputs, dict):
            actions = model_outputs.get("actions")
        elif isinstance(model_outputs, torch.Tensor):
            actions = model_outputs
        else:
            actions = model_outputs[0]
        
        if actions is None:
            raise ValueError("Could not extract actions from model outputs")
        
        # Ensure correct shape
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        return actions.cpu()


class EnsemblePolicy(BasePolicy):
    """
    Ensemble of multiple policies for robust predictions.
    
    Combines predictions from multiple models using voting or averaging.
    """
    
    def __init__(
        self,
        policies: List[BasePolicy],
        aggregation: str = "mean",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize ensemble policy.
        
        Args:
            policies: List of policies to ensemble
            aggregation: Aggregation method ("mean", "median", "vote")
            device: Device for inference
        """
        super().__init__(model=None, device=device)
        
        if not policies:
            raise ValueError("At least one policy required for ensemble")
        
        self.policies = policies
        self.aggregation = aggregation
        self._is_loaded = all(p.is_loaded for p in policies)
    
    @torch.no_grad()
    def predict(
        self,
        observation: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ensemble prediction.
        
        Args:
            observation: Observation dictionary
            **kwargs: Additional arguments
            
        Returns:
            Aggregated predictions from all policies
        """
        if not self.is_loaded:
            raise RuntimeError("Not all policies are loaded")
        
        # Collect predictions from all policies
        predictions = []
        for policy in self.policies:
            pred = policy.predict(observation, **kwargs)
            predictions.append(pred["actions"])
        
        # Stack predictions
        actions_stack = torch.stack(predictions, dim=0)
        
        # Aggregate
        if self.aggregation == "mean":
            actions = actions_stack.mean(dim=0)
        elif self.aggregation == "median":
            actions = actions_stack.median(dim=0)[0]
        elif self.aggregation == "vote":
            # TODO: Implement voting for discrete actions
            actions = actions_stack.mode(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return {
            "actions": actions,
            "metadata": {
                "num_policies": len(self.policies),
                "aggregation": self.aggregation,
            }
        }
    
    def reset(self) -> None:
        """Reset all policies in ensemble."""
        for policy in self.policies:
            policy.reset()
