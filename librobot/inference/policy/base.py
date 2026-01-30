"""Policy wrappers for VLA inference."""

from typing import Any, Dict, List, Optional, Union
import numpy as np


class BasePolicy:
    """Base policy wrapper for VLA models."""
    
    def __init__(
        self,
        model: Any,
        action_dim: int = 7,
        action_horizon: int = 1,
        device: str = "cuda",
    ):
        """
        Args:
            model: VLA model instance
            action_dim: Action dimension
            action_horizon: Number of actions to predict
            device: Inference device
        """
        self.model = model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = device
        
        # Put model in eval mode
        if hasattr(model, 'eval'):
            model.eval()
    
    def __call__(
        self,
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> np.ndarray:
        """
        Get action from observation.
        
        Args:
            observation: Observation dict with 'images', 'proprioception'
            instruction: Language instruction
            
        Returns:
            Action array [action_dim] or [action_horizon, action_dim]
        """
        return self.get_action(observation, instruction)
    
    def get_action(
        self,
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> np.ndarray:
        """Get action from observation."""
        try:
            import torch
            
            with torch.no_grad():
                # Prepare inputs
                inputs = self._prepare_inputs(observation, instruction)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Extract action
                action = self._extract_action(outputs)
                
                return action.cpu().numpy()
                
        except ImportError:
            return np.zeros(self.action_dim)
    
    def _prepare_inputs(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """Prepare model inputs."""
        import torch
        
        inputs = {}
        
        # Process images
        if 'images' in observation:
            images = observation['images']
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            if images.dim() == 3:
                images = images.unsqueeze(0)  # Add batch dim
            inputs['images'] = images.to(self.device)
        
        # Process proprioception
        if 'proprioception' in observation:
            state = observation['proprioception']
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            inputs['proprioception'] = state.to(self.device)
        
        # Add instruction
        inputs['text'] = instruction
        
        return inputs
    
    def _extract_action(self, outputs: Dict[str, Any]) -> Any:
        """Extract action from model outputs."""
        if isinstance(outputs, dict):
            if 'actions' in outputs:
                return outputs['actions'][0]  # First in batch
            if 'action' in outputs:
                return outputs['action'][0]
        return outputs[0] if hasattr(outputs, '__getitem__') else outputs
    
    def reset(self) -> None:
        """Reset policy state (for recurrent policies)."""
        pass
    
    def predict(
        self,
        observation: Dict[str, Any],
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict actions from observation.
        
        Args:
            observation: Observation dict with images, state, text
            return_logits: Whether to return logits
            
        Returns:
            Dict with 'actions' and optional 'metadata'
        """
        instruction = observation.get("text", "")
        action = self.get_action(observation, instruction)
        
        return {
            "actions": action,
            "metadata": {
                "action_dim": self.action_dim,
                "action_horizon": self.action_horizon,
            }
        }


class DiffusionPolicy(BasePolicy):
    """Policy wrapper for diffusion-based models."""
    
    def __init__(
        self,
        model: Any,
        action_dim: int = 7,
        action_horizon: int = 10,
        num_inference_steps: int = 10,
        device: str = "cuda",
    ):
        super().__init__(model, action_dim, action_horizon, device)
        self.num_inference_steps = num_inference_steps
    
    def get_action(
        self,
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> np.ndarray:
        """Get action using diffusion sampling."""
        try:
            import torch
            
            with torch.no_grad():
                inputs = self._prepare_inputs(observation, instruction)
                
                # Sample from diffusion
                if hasattr(self.model, 'sample'):
                    actions = self.model.sample(
                        **inputs,
                        num_steps=self.num_inference_steps,
                    )
                else:
                    outputs = self.model(**inputs)
                    actions = self._extract_action(outputs)
                
                return actions.cpu().numpy()
                
        except ImportError:
            return np.zeros((self.action_horizon, self.action_dim))


class AutoregressivePolicy(BasePolicy):
    """Policy wrapper for autoregressive token-based models."""
    
    def __init__(
        self,
        model: Any,
        action_dim: int = 7,
        num_bins: int = 256,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(model, action_dim, 1, device)
        self.num_bins = num_bins
        self.temperature = temperature
    
    def get_action(
        self,
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> np.ndarray:
        """Get action via autoregressive generation."""
        try:
            import torch
            
            with torch.no_grad():
                inputs = self._prepare_inputs(observation, instruction)
                
                if hasattr(self.model, 'generate_action'):
                    action = self.model.generate_action(
                        **inputs,
                        temperature=self.temperature,
                    )
                else:
                    outputs = self.model(**inputs)
                    action = self._extract_action(outputs)
                
                return action.cpu().numpy()
                
        except ImportError:
            return np.zeros(self.action_dim)


class EnsemblePolicy(BasePolicy):
    """Ensemble of multiple policies."""
    
    def __init__(
        self,
        policies: List[BasePolicy],
        aggregation: str = "mean",
    ):
        """
        Args:
            policies: List of policy instances
            aggregation: How to combine predictions ("mean", "median", "vote")
        """
        self.policies = policies
        self.aggregation = aggregation
        self.action_dim = policies[0].action_dim if policies else 7
    
    def get_action(
        self,
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> np.ndarray:
        """Get ensemble action."""
        actions = [p.get_action(observation, instruction) for p in self.policies]
        actions = np.stack(actions)
        
        if self.aggregation == "mean":
            return np.mean(actions, axis=0)
        elif self.aggregation == "median":
            return np.median(actions, axis=0)
        else:
            return actions[0]  # Default to first
    
    def reset(self) -> None:
        for p in self.policies:
            p.reset()


__all__ = [
    'BasePolicy',
    'DiffusionPolicy',
    'AutoregressivePolicy',
    'EnsemblePolicy',
]
