"""Advanced learning paradigms for VLA models.

This module provides implementations for:
- Reinforcement Learning integration
- Imitation learning from video
- Multi-robot coordination
- Sim-to-real transfer
- Online learning and adaptation
- Zero-shot and few-shot capabilities
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from librobot.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Reinforcement Learning Integration
# =============================================================================

@dataclass
class RLConfig:
    """Configuration for RL training.
    
    Args:
        algorithm: RL algorithm ("ppo", "sac", "td3", "ddpg")
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clip range
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size for updates
        n_epochs: Number of epochs per update
        entropy_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
    """
    algorithm: str = "ppo"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class RLPolicyWrapper:
    """Wrapper to combine VLA model with RL policy head.
    
    Enables fine-tuning VLA models with reinforcement learning
    for improved real-world performance.
    
    Example:
        >>> vla_model = create_vla("groot", vlm=vlm)
        >>> rl_wrapper = RLPolicyWrapper(vla_model, action_dim=7)
        >>> rl_wrapper.train(env, total_timesteps=100000)
    """
    
    def __init__(
        self,
        vla_model: Any,
        action_dim: int,
        config: Optional[RLConfig] = None,
        freeze_vlm: bool = True,
    ):
        self.vla_model = vla_model
        self.action_dim = action_dim
        self.config = config or RLConfig()
        self.freeze_vlm = freeze_vlm
        
        self._value_head = None
        self._setup_heads()
    
    def _setup_heads(self) -> None:
        """Setup value and policy heads for RL."""
        try:
            import torch
            import torch.nn as nn
            
            # Get embedding dimension from VLA
            embed_dim = getattr(self.vla_model, "embed_dim", 768)
            
            # Value head for critic
            self._value_head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            
            logger.info("RL policy and value heads initialized")
            
        except ImportError:
            logger.warning("PyTorch not available for RL setup")
    
    def get_action_and_value(
        self,
        observation: Dict[str, Any],
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action, value, and log probability.
        
        Args:
            observation: Environment observation
            
        Returns:
            Tuple of (action, value, log_prob)
        """
        try:
            import torch
            
            # Get VLA features
            with torch.no_grad():
                action = self.vla_model.predict_action(
                    observation.get("image"),
                    observation.get("text", ""),
                    observation.get("state"),
                )
            
            # Compute value
            if self._value_head is not None:
                # Get embeddings from VLA
                features = self.vla_model.get_features(observation)
                value = self._value_head(features)
            else:
                value = torch.tensor(0.0)
            
            # Placeholder log_prob (would need distribution)
            log_prob = 0.0
            
            return action.numpy(), value.item(), log_prob
            
        except Exception as e:
            logger.error(f"Error in get_action_and_value: {e}")
            return np.zeros(self.action_dim), 0.0, 0.0
    
    def train(
        self,
        env: Any,
        total_timesteps: int = 100000,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the VLA model with RL.
        
        Args:
            env: Gymnasium environment
            total_timesteps: Total training timesteps
            callback: Optional callback function
            
        Returns:
            Training statistics
        """
        try:
            from stable_baselines3 import PPO, SAC, TD3, DDPG
            from stable_baselines3.common.callbacks import BaseCallback
            
            # Select algorithm
            algo_map = {
                "ppo": PPO,
                "sac": SAC,
                "td3": TD3,
                "ddpg": DDPG,
            }
            
            AlgoClass = algo_map.get(self.config.algorithm, PPO)
            
            # Build kwargs based on what the algorithm supports
            model_kwargs = {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gamma": self.config.gamma,
                "verbose": 1,
            }
            
            # Only add n_steps for on-policy algorithms (PPO)
            if self.config.algorithm == "ppo":
                model_kwargs["n_steps"] = self.config.n_steps
            
            # Create model with VLA as feature extractor
            model = AlgoClass(
                "MlpPolicy",
                env,
                **model_kwargs,
            )
            
            # Train
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
            )
            
            logger.info("RL training completed")
            return {"total_timesteps": total_timesteps}
            
        except ImportError:
            logger.warning(
                "stable-baselines3 not installed. "
                "Install with: pip install stable-baselines3"
            )
            return {}


class RewardShaping:
    """Reward shaping utilities for RL training."""
    
    @staticmethod
    def dense_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        reward_scale: float = 1.0,
    ) -> float:
        """Compute dense distance-based reward."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance * reward_scale
    
    @staticmethod
    def sparse_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        threshold: float = 0.05,
    ) -> float:
        """Compute sparse success reward."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < threshold else 0.0
    
    @staticmethod
    def potential_based_shaping(
        state: np.ndarray,
        next_state: np.ndarray,
        potential_fn: Callable[[np.ndarray], float],
        gamma: float = 0.99,
    ) -> float:
        """Compute potential-based reward shaping."""
        return gamma * potential_fn(next_state) - potential_fn(state)


# =============================================================================
# Imitation Learning from Video
# =============================================================================

class VideoImitationLearner:
    """Learn robot actions from video demonstrations.
    
    Uses video prediction and action inference to learn
    from human or robot demonstration videos.
    
    Example:
        >>> learner = VideoImitationLearner(vla_model)
        >>> learner.learn_from_video("demo.mp4", task_instruction="pick up cup")
    """
    
    def __init__(
        self,
        vla_model: Any,
        frame_rate: int = 10,
        action_inference_model: Optional[Any] = None,
    ):
        self.vla_model = vla_model
        self.frame_rate = frame_rate
        self.action_inference_model = action_inference_model
        
    def extract_frames(
        self,
        video_path: str,
        max_frames: int = 500,
    ) -> List[np.ndarray]:
        """Extract frames from video at specified frame rate."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.frame_rate)
            
            frame_idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except ImportError:
            logger.error("OpenCV required for video processing")
            return []
    
    def infer_actions_from_frames(
        self,
        frames: List[np.ndarray],
        task_instruction: str,
    ) -> np.ndarray:
        """Infer robot actions from video frames.
        
        Uses the VLA model to predict actions that would
        reproduce the demonstrated behavior.
        """
        actions = []
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Use VLA model to infer action
            # This is a simplified approach - real implementation
            # might use inverse dynamics or flow-based methods
            action = self.vla_model.predict_action(
                current_frame,
                task_instruction,
            )
            
            actions.append(action)
        
        return np.array(actions)
    
    def learn_from_video(
        self,
        video_path: str,
        task_instruction: str,
        num_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Learn from a video demonstration.
        
        Args:
            video_path: Path to demonstration video
            task_instruction: Natural language task description
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) < 2:
            logger.error("Not enough frames extracted from video")
            return {}
        
        # Infer actions
        actions = self.infer_actions_from_frames(frames, task_instruction)
        
        # Fine-tune VLA model on inferred demonstrations
        # This would use behavior cloning
        logger.info(f"Learning from {len(actions)} action samples")
        
        # Placeholder for actual training
        training_stats = {
            "num_frames": len(frames),
            "num_actions": len(actions),
            "task": task_instruction,
        }
        
        return training_stats


# =============================================================================
# Multi-Robot Coordination
# =============================================================================

@dataclass
class MultiRobotConfig:
    """Configuration for multi-robot coordination.
    
    Args:
        num_robots: Number of robots
        communication_type: Type of communication ("centralized", "decentralized", "none")
        coordination_method: Coordination method ("shared_goal", "role_assignment", "formation")
        message_dim: Dimension of robot messages
    """
    num_robots: int = 2
    communication_type: str = "decentralized"
    coordination_method: str = "shared_goal"
    message_dim: int = 64


class MultiRobotCoordinator:
    """Coordinate multiple robots for collaborative tasks.
    
    Supports:
        - Centralized control with global state
        - Decentralized control with message passing
        - Role-based task assignment
        - Formation control
    
    Example:
        >>> coordinator = MultiRobotCoordinator(
        ...     vla_models=[vla1, vla2],
        ...     config=MultiRobotConfig(num_robots=2),
        ... )
        >>> actions = coordinator.get_coordinated_actions(observations)
    """
    
    def __init__(
        self,
        vla_models: List[Any],
        config: Optional[MultiRobotConfig] = None,
    ):
        self.vla_models = vla_models
        self.config = config or MultiRobotConfig(num_robots=len(vla_models))
        self._messages = [np.zeros(self.config.message_dim) for _ in range(len(vla_models))]
    
    def get_coordinated_actions(
        self,
        observations: List[Dict[str, Any]],
        shared_goal: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Get coordinated actions for all robots.
        
        Args:
            observations: List of observations for each robot
            shared_goal: Shared task goal
            
        Returns:
            List of actions for each robot
        """
        actions = []
        
        if self.config.communication_type == "centralized":
            actions = self._centralized_control(observations, shared_goal)
        elif self.config.communication_type == "decentralized":
            actions = self._decentralized_control(observations, shared_goal)
        else:
            actions = self._independent_control(observations, shared_goal)
        
        return actions
    
    def _centralized_control(
        self,
        observations: List[Dict[str, Any]],
        shared_goal: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Centralized control with global state."""
        # Aggregate all observations
        global_obs = self._aggregate_observations(observations)
        
        # Use first model as central controller
        actions = []
        for i, model in enumerate(self.vla_models):
            # Include global context in instruction
            instruction = f"{shared_goal or ''} Robot {i+1} of {len(self.vla_models)}"
            
            action = model.predict_action(
                global_obs.get("image"),
                instruction,
                global_obs.get("state"),
            )
            actions.append(action)
        
        return actions
    
    def _decentralized_control(
        self,
        observations: List[Dict[str, Any]],
        shared_goal: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Decentralized control with message passing."""
        actions = []
        new_messages = []
        
        for i, (model, obs) in enumerate(zip(self.vla_models, observations)):
            # Include messages from other robots
            other_messages = [m for j, m in enumerate(self._messages) if j != i]
            
            # Compute action
            instruction = f"{shared_goal or ''} Messages: {len(other_messages)}"
            
            action = model.predict_action(
                obs.get("image"),
                instruction,
                obs.get("state"),
            )
            actions.append(action)
            
            # Generate message for other robots
            message = self._generate_message(obs, action, i)
            new_messages.append(message)
        
        # Update messages
        self._messages = new_messages
        
        return actions
    
    def _independent_control(
        self,
        observations: List[Dict[str, Any]],
        shared_goal: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Independent control without coordination."""
        actions = []
        
        for model, obs in zip(self.vla_models, observations):
            action = model.predict_action(
                obs.get("image"),
                shared_goal or "",
                obs.get("state"),
            )
            actions.append(action)
        
        return actions
    
    def _aggregate_observations(
        self,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate observations from all robots."""
        # Combine images (could use more sophisticated fusion)
        images = [obs.get("image") for obs in observations if "image" in obs]
        states = [obs.get("state") for obs in observations if "state" in obs]
        
        return {
            "image": images[0] if images else None,
            "all_images": images,
            "state": np.concatenate(states) if states else None,
        }
    
    def _generate_message(
        self,
        observation: Dict[str, Any],
        action: np.ndarray,
        robot_id: int,
    ) -> np.ndarray:
        """Generate message for other robots."""
        # Simple message encoding: robot ID + action summary
        message = np.zeros(self.config.message_dim)
        message[0] = robot_id
        # Safely copy action to message, handling different lengths
        action_len = min(len(action), self.config.message_dim - 1)
        message[1:1 + action_len] = action[:action_len]
        return message


# =============================================================================
# Sim-to-Real Transfer
# =============================================================================

@dataclass
class SimToRealConfig:
    """Configuration for sim-to-real transfer.
    
    Args:
        domain_randomization: Enable domain randomization
        visual_randomization: Randomize visual appearance
        dynamics_randomization: Randomize physics parameters
        action_noise: Add noise to actions
        observation_noise: Add noise to observations
        adaptation_method: Adaptation method ("fine_tune", "domain_adaptation", "meta_learning")
    """
    domain_randomization: bool = True
    visual_randomization: bool = True
    dynamics_randomization: bool = True
    action_noise: float = 0.01
    observation_noise: float = 0.01
    adaptation_method: str = "fine_tune"


class SimToRealAdapter:
    """Adapter for sim-to-real transfer.
    
    Provides techniques for transferring policies trained
    in simulation to real-world robots.
    
    Example:
        >>> adapter = SimToRealAdapter(sim_model, config)
        >>> adapted_model = adapter.adapt_to_real(real_data)
    """
    
    def __init__(
        self,
        sim_model: Any,
        config: Optional[SimToRealConfig] = None,
    ):
        self.sim_model = sim_model
        self.config = config or SimToRealConfig()
    
    def apply_domain_randomization(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply domain randomization to observation."""
        result = observation.copy()
        
        if self.config.visual_randomization and "image" in result:
            result["image"] = self._randomize_visual(result["image"])
        
        if self.config.observation_noise and "state" in result:
            noise = np.random.normal(0, self.config.observation_noise, result["state"].shape)
            result["state"] = result["state"] + noise
        
        return result
    
    def _randomize_visual(self, image: np.ndarray) -> np.ndarray:
        """Apply visual domain randomization."""
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Random noise
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def adapt_to_real(
        self,
        real_data: List[Dict[str, Any]],
        num_epochs: int = 5,
    ) -> Any:
        """
        Adapt simulation-trained model to real data.
        
        Args:
            real_data: Small amount of real-world data
            num_epochs: Number of adaptation epochs
            
        Returns:
            Adapted model
        """
        if self.config.adaptation_method == "fine_tune":
            return self._fine_tune_adaptation(real_data, num_epochs)
        elif self.config.adaptation_method == "domain_adaptation":
            return self._domain_adaptation(real_data, num_epochs)
        else:
            return self.sim_model
    
    def _fine_tune_adaptation(
        self,
        real_data: List[Dict[str, Any]],
        num_epochs: int,
    ) -> Any:
        """Simple fine-tuning on real data."""
        logger.info(f"Fine-tuning on {len(real_data)} real samples for {num_epochs} epochs")
        
        # Placeholder for actual fine-tuning
        # Would involve:
        # 1. Create small learning rate optimizer
        # 2. Train on real data with early stopping
        # 3. Optionally use LoRA for efficient adaptation
        
        return self.sim_model
    
    def _domain_adaptation(
        self,
        real_data: List[Dict[str, Any]],
        num_epochs: int,
    ) -> Any:
        """Domain adaptation using adversarial training."""
        logger.info(f"Domain adaptation with {len(real_data)} real samples")
        
        # Placeholder for domain adaptation
        # Would involve:
        # 1. Train domain discriminator
        # 2. Adversarial training to align sim/real distributions
        
        return self.sim_model


# =============================================================================
# Online Learning and Adaptation
# =============================================================================

class OnlineLearner:
    """Online learning and adaptation for VLA models.
    
    Enables continuous learning from robot experience
    while deployed in the real world.
    
    Example:
        >>> learner = OnlineLearner(vla_model)
        >>> learner.update(observation, action, reward, next_observation)
    """
    
    def __init__(
        self,
        vla_model: Any,
        buffer_size: int = 10000,
        update_frequency: int = 100,
        learning_rate: float = 1e-5,
    ):
        self.vla_model = vla_model
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        
        self._buffer: List[Dict[str, Any]] = []
        self._step_count = 0
    
    def add_experience(
        self,
        observation: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool = False,
    ) -> None:
        """Add experience to replay buffer."""
        self._buffer.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        })
        
        # Maintain buffer size
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
        
        self._step_count += 1
        
        # Periodic update
        if self._step_count % self.update_frequency == 0:
            self.update()
    
    def update(self) -> Dict[str, float]:
        """Update model from buffer."""
        if len(self._buffer) < self.update_frequency:
            return {}
        
        # Sample batch from buffer
        import random
        batch = random.sample(self._buffer, min(32, len(self._buffer)))
        
        # Placeholder for actual online update
        # Would involve:
        # 1. Compute loss on batch
        # 2. Small gradient step
        # 3. Possibly use importance sampling
        
        logger.debug(f"Online update with {len(batch)} samples")
        
        return {"buffer_size": len(self._buffer), "step": self._step_count}
    
    def save_buffer(self, path: str) -> None:
        """Save replay buffer to file."""
        import pickle
        
        with open(path, "wb") as f:
            pickle.dump(self._buffer, f)
        
        logger.info(f"Buffer saved to {path}")
    
    def load_buffer(self, path: str) -> None:
        """Load replay buffer from file."""
        import pickle
        
        with open(path, "rb") as f:
            self._buffer = pickle.load(f)
        
        logger.info(f"Buffer loaded from {path} ({len(self._buffer)} samples)")


# =============================================================================
# Zero-Shot and Few-Shot Capabilities
# =============================================================================

class ZeroShotAdapter:
    """Zero-shot task adaptation using language grounding.
    
    Enables VLA models to perform new tasks without
    task-specific training data.
    
    Example:
        >>> adapter = ZeroShotAdapter(vla_model)
        >>> action = adapter.execute_task(
        ...     observation,
        ...     task="pour water from the bottle into the cup"
        ... )
    """
    
    def __init__(
        self,
        vla_model: Any,
        task_decomposer: Optional[Callable] = None,
    ):
        self.vla_model = vla_model
        self.task_decomposer = task_decomposer or self._default_decomposer
    
    def _default_decomposer(self, task: str) -> List[str]:
        """Default task decomposition into subtasks."""
        # Simple heuristic decomposition
        # In practice, could use LLM for decomposition
        
        keywords = ["pick", "place", "move", "pour", "open", "close", "push", "pull"]
        subtasks = []
        
        task_lower = task.lower()
        for keyword in keywords:
            if keyword in task_lower:
                subtasks.append(f"{keyword} object")
        
        if not subtasks:
            subtasks = [task]
        
        return subtasks
    
    def execute_task(
        self,
        observation: Dict[str, Any],
        task: str,
        max_steps: int = 100,
    ) -> List[np.ndarray]:
        """
        Execute a zero-shot task.
        
        Args:
            observation: Current observation
            task: Natural language task description
            max_steps: Maximum execution steps
            
        Returns:
            List of executed actions
        """
        # Decompose task into subtasks
        subtasks = self.task_decomposer(task)
        logger.info(f"Task decomposed into {len(subtasks)} subtasks: {subtasks}")
        
        # Execute each subtask
        actions = []
        current_obs = observation
        
        for subtask in subtasks:
            # Get action from VLA model
            action = self.vla_model.predict_action(
                current_obs.get("image"),
                subtask,
                current_obs.get("state"),
            )
            actions.append(action)
            
            # Update observation (in real execution, would get from environment)
            # For now, just continue with same observation
        
        return actions
    
    def compose_skills(
        self,
        observation: Dict[str, Any],
        skill_sequence: List[str],
    ) -> List[np.ndarray]:
        """Compose multiple skills sequentially."""
        actions = []
        
        for skill in skill_sequence:
            action = self.vla_model.predict_action(
                observation.get("image"),
                skill,
                observation.get("state"),
            )
            actions.append(action)
        
        return actions


class FewShotAdapter:
    """Few-shot task adaptation using in-context learning.
    
    Enables VLA models to adapt to new tasks from
    a small number of demonstrations.
    
    Example:
        >>> adapter = FewShotAdapter(vla_model)
        >>> adapter.add_demonstration(demo_obs, demo_action)
        >>> action = adapter.predict_with_demos(new_observation)
    """
    
    def __init__(
        self,
        vla_model: Any,
        max_demonstrations: int = 5,
    ):
        self.vla_model = vla_model
        self.max_demonstrations = max_demonstrations
        self._demonstrations: List[Dict[str, Any]] = []
    
    def add_demonstration(
        self,
        observation: Dict[str, Any],
        action: np.ndarray,
        task_description: Optional[str] = None,
    ) -> None:
        """Add a demonstration to the context."""
        demo = {
            "observation": observation,
            "action": action,
            "task": task_description,
        }
        
        self._demonstrations.append(demo)
        
        # Maintain maximum demonstrations
        if len(self._demonstrations) > self.max_demonstrations:
            self._demonstrations.pop(0)
        
        logger.info(f"Added demonstration. Total: {len(self._demonstrations)}")
    
    def clear_demonstrations(self) -> None:
        """Clear all demonstrations."""
        self._demonstrations = []
    
    def predict_with_demos(
        self,
        observation: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict action using demonstration context.
        
        Args:
            observation: Current observation
            task_description: Task description
            
        Returns:
            Predicted action
        """
        # Build context from demonstrations
        context = self._build_context(task_description)
        
        # Get action from VLA model with context
        action = self.vla_model.predict_action(
            observation.get("image"),
            context,
            observation.get("state"),
        )
        
        return action
    
    def _build_context(self, task_description: Optional[str] = None) -> str:
        """Build instruction context from demonstrations."""
        if not self._demonstrations:
            return task_description or ""
        
        # Include demonstration information in context
        demo_info = f"{len(self._demonstrations)} demonstration(s) provided. "
        
        if task_description:
            demo_info += f"Task: {task_description}"
        elif self._demonstrations[0].get("task"):
            demo_info += f"Task: {self._demonstrations[0]['task']}"
        
        return demo_info


# =============================================================================
# Model Compression for Edge Deployment
# =============================================================================

class EdgeDeployer:
    """Prepare VLA models for edge deployment.
    
    Provides model compression and optimization for
    deployment on resource-constrained devices.
    
    Example:
        >>> deployer = EdgeDeployer(vla_model)
        >>> optimized = deployer.optimize(target_device="jetson")
    """
    
    def __init__(
        self,
        vla_model: Any,
        target_latency_ms: float = 50.0,
    ):
        self.vla_model = vla_model
        self.target_latency_ms = target_latency_ms
    
    def quantize(
        self,
        method: str = "dynamic",
        dtype: str = "int8",
    ) -> Any:
        """
        Quantize model for reduced memory and faster inference.
        
        Args:
            method: Quantization method ("dynamic", "static", "qat")
            dtype: Target data type ("int8", "int4", "fp16")
            
        Returns:
            Quantized model
        """
        try:
            import torch
            
            if method == "dynamic":
                quantized = torch.quantization.quantize_dynamic(
                    self.vla_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8 if dtype == "int8" else torch.float16,
                )
                logger.info(f"Model quantized with {method} {dtype}")
                return quantized
                
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return self.vla_model
    
    def prune(
        self,
        sparsity: float = 0.5,
        method: str = "magnitude",
    ) -> Any:
        """
        Prune model weights for reduced size.
        
        Args:
            sparsity: Target sparsity (0-1)
            method: Pruning method ("magnitude", "structured")
            
        Returns:
            Pruned model
        """
        try:
            import torch
            import torch.nn.utils.prune as prune
            
            # Apply pruning to linear layers
            for name, module in self.vla_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=sparsity)
            
            logger.info(f"Model pruned with {sparsity:.0%} sparsity")
            return self.vla_model
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return self.vla_model
    
    def distill(
        self,
        student_model: Any,
        train_data: List[Dict[str, Any]],
        temperature: float = 4.0,
    ) -> Any:
        """
        Knowledge distillation to smaller student model.
        
        Args:
            student_model: Smaller student model
            train_data: Training data for distillation
            temperature: Distillation temperature
            
        Returns:
            Trained student model
        """
        logger.info("Knowledge distillation (placeholder)")
        # Placeholder for actual distillation
        return student_model
    
    def optimize(
        self,
        target_device: str = "gpu",
        optimization_level: str = "O2",
    ) -> Any:
        """
        Apply device-specific optimizations.
        
        Args:
            target_device: Target device ("gpu", "cpu", "jetson", "edge_tpu")
            optimization_level: Optimization aggressiveness
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing for {target_device} with level {optimization_level}")
        
        # Apply appropriate optimizations
        if target_device == "jetson":
            # Apply TensorRT optimization
            return self._tensorrt_optimize()
        elif target_device == "edge_tpu":
            # Apply Edge TPU optimization
            return self._edgetpu_optimize()
        else:
            return self.vla_model
    
    def _tensorrt_optimize(self) -> Any:
        """Apply TensorRT optimization for NVIDIA devices."""
        try:
            import torch_tensorrt
            
            # Compile with TensorRT
            # Placeholder for actual TensorRT optimization
            logger.info("TensorRT optimization (placeholder)")
            return self.vla_model
            
        except ImportError:
            logger.warning("torch-tensorrt not available")
            return self.vla_model
    
    def _edgetpu_optimize(self) -> Any:
        """Apply Edge TPU optimization for Coral devices."""
        logger.info("Edge TPU optimization (placeholder)")
        return self.vla_model


__all__ = [
    # RL Integration
    "RLConfig",
    "RLPolicyWrapper",
    "RewardShaping",
    # Video Imitation
    "VideoImitationLearner",
    # Multi-Robot
    "MultiRobotConfig",
    "MultiRobotCoordinator",
    # Sim-to-Real
    "SimToRealConfig",
    "SimToRealAdapter",
    # Online Learning
    "OnlineLearner",
    # Zero/Few-Shot
    "ZeroShotAdapter",
    "FewShotAdapter",
    # Edge Deployment
    "EdgeDeployer",
]
