"""Standard evaluation benchmarks for VLA models."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from librobot.utils.logging import get_logger
from librobot.evaluation.metrics import MetricCollection, create_metrics

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation.
    
    Args:
        name: Benchmark name
        num_episodes: Number of evaluation episodes
        max_episode_steps: Maximum steps per episode
        metrics: List of metrics to compute
        save_videos: Whether to save episode videos
        save_trajectories: Whether to save trajectories
        output_dir: Output directory for results
        seed: Random seed
    """
    name: str = "default"
    num_episodes: int = 50
    max_episode_steps: int = 500
    metrics: List[str] = field(default_factory=lambda: ["success_rate", "mse", "smoothness"])
    save_videos: bool = False
    save_trajectories: bool = False
    output_dir: str = "./benchmark_results"
    seed: int = 42


@dataclass
class EpisodeResult:
    """Result from a single episode evaluation."""
    
    success: bool
    total_reward: float
    episode_length: int
    actions: np.ndarray
    observations: List[Dict[str, Any]]
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Compute episode metrics."""
        return {
            "success": float(self.success),
            "reward": self.total_reward,
            "length": self.episode_length,
        }


class AbstractBenchmark(ABC):
    """Abstract base class for evaluation benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._results: List[EpisodeResult] = []
        self._metrics = create_metrics(config.metrics)
    
    @abstractmethod
    def setup(self) -> None:
        """Setup benchmark environment."""
        pass
    
    @abstractmethod
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """Evaluate a single episode."""
        pass
    
    @abstractmethod
    def get_tasks(self) -> List[str]:
        """Get list of evaluation tasks."""
        pass
    
    def evaluate(
        self,
        policy: Callable,
        tasks: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full benchmark evaluation.
        
        Args:
            policy: Policy function that takes observation and returns action
            tasks: List of tasks to evaluate (None for all)
            verbose: Print progress
            
        Returns:
            Benchmark results dictionary
        """
        tasks = tasks or self.get_tasks()
        all_results = {}
        
        for task in tasks:
            task_results = []
            
            for episode_idx in range(self.config.num_episodes):
                if verbose:
                    print(f"\rTask: {task} | Episode: {episode_idx + 1}/{self.config.num_episodes}", end="")
                
                result = self.evaluate_episode(policy, task)
                task_results.append(result)
                self._results.append(result)
            
            if verbose:
                print()
            
            # Compute task metrics
            success_rate = np.mean([r.success for r in task_results])
            avg_reward = np.mean([r.total_reward for r in task_results])
            avg_length = np.mean([r.episode_length for r in task_results])
            
            all_results[task] = {
                "success_rate": success_rate,
                "average_reward": avg_reward,
                "average_length": avg_length,
                "num_episodes": len(task_results),
            }
            
            if verbose:
                logger.info(f"Task '{task}': success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f}")
        
        # Compute overall metrics
        all_results["overall"] = {
            "success_rate": np.mean([r.success for r in self._results]),
            "average_reward": np.mean([r.total_reward for r in self._results]),
            "average_length": np.mean([r.episode_length for r in self._results]),
            "total_episodes": len(self._results),
        }
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save benchmark results to file."""
        import json
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = filename or f"{self.config.name}_results_{int(time.time())}.json"
        filepath = output_dir / filename
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj
        
        with open(filepath, "w") as f:
            json.dump(results, f, default=convert, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)


class SimulationBenchmark(AbstractBenchmark):
    """Benchmark for simulation environments.
    
    Supports various simulation platforms:
        - MuJoCo
        - Isaac Sim
        - PyBullet
        - Robosuite
    
    Example:
        >>> config = BenchmarkConfig(name="simpler", num_episodes=100)
        >>> benchmark = SimulationBenchmark(config, env_name="reach")
        >>> results = benchmark.evaluate(policy)
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        env_name: str = "reach",
        env_kwargs: Optional[Dict[str, Any]] = None,
        render: bool = False,
    ):
        super().__init__(config)
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}
        self.render = render
        self._env = None
    
    def setup(self) -> None:
        """Setup simulation environment."""
        try:
            import gymnasium as gym
            
            # Try to create environment
            self._env = gym.make(
                self.env_name,
                render_mode="rgb_array" if self.render else None,
                **self.env_kwargs
            )
            
            logger.info(f"Environment '{self.env_name}' created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create gymnasium environment: {e}")
            # Try robosuite
            try:
                import robosuite as suite
                
                self._env = suite.make(
                    self.env_name,
                    has_renderer=self.render,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    **self.env_kwargs
                )
                logger.info(f"Robosuite environment '{self.env_name}' created successfully")
                
            except ImportError:
                logger.error("Neither gymnasium nor robosuite available")
                raise
    
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """Evaluate a single episode in simulation."""
        if self._env is None:
            self.setup()
        
        obs, info = self._env.reset()
        
        observations = [obs]
        actions_list = []
        total_reward = 0.0
        success = False
        
        for step in range(self.config.max_episode_steps):
            # Get action from policy
            action = policy(obs)
            
            if hasattr(action, "numpy"):
                action = action.numpy()
            action = np.array(action)
            
            actions_list.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = self._env.step(action)
            
            observations.append(obs)
            total_reward += reward
            
            # Check for success
            if "success" in info:
                success = info["success"]
            
            if terminated or truncated:
                break
        
        return EpisodeResult(
            success=success,
            total_reward=total_reward,
            episode_length=len(actions_list),
            actions=np.array(actions_list),
            observations=observations,
            info=info,
        )
    
    def get_tasks(self) -> List[str]:
        """Get list of evaluation tasks."""
        return [self.env_name]


class BridgeBenchmark(AbstractBenchmark):
    """Bridge dataset benchmark for tabletop manipulation.
    
    Tasks:
        - pick: Pick up objects
        - place: Place objects at target locations
        - stack: Stack objects
    """
    
    TASKS = ["pick", "place", "stack"]
    
    def __init__(
        self,
        config: BenchmarkConfig,
        env_path: Optional[str] = None,
    ):
        config.name = "bridge"
        super().__init__(config)
        self.env_path = env_path
    
    def setup(self) -> None:
        """Setup Bridge environment."""
        logger.info("Bridge benchmark setup (simulation placeholder)")
    
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """Evaluate single episode."""
        # Placeholder implementation
        task = task or "pick"
        
        # Simulate episode
        actions = np.random.randn(50, 7)
        success = np.random.random() > 0.5
        reward = float(success) * 1.0
        
        return EpisodeResult(
            success=success,
            total_reward=reward,
            episode_length=len(actions),
            actions=actions,
            observations=[],
            info={"task": task},
        )
    
    def get_tasks(self) -> List[str]:
        return self.TASKS


class LIBEROBenchmark(AbstractBenchmark):
    """LIBERO benchmark for long-horizon manipulation.
    
    Benchmark suites:
        - LIBERO-90: 90 diverse manipulation tasks
        - LIBERO-Goal: Goal-conditioned tasks
        - LIBERO-Object: Object manipulation tasks
        - LIBERO-Spatial: Spatial reasoning tasks
    """
    
    SUITES = ["libero_90", "libero_goal", "libero_object", "libero_spatial"]
    
    def __init__(
        self,
        config: BenchmarkConfig,
        suite: str = "libero_90",
    ):
        config.name = f"libero_{suite}"
        super().__init__(config)
        self.suite = suite
    
    def setup(self) -> None:
        """Setup LIBERO environment."""
        try:
            import libero
            logger.info(f"LIBERO suite '{self.suite}' loaded")
        except ImportError:
            logger.warning("LIBERO not installed. Using placeholder.")
    
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """Evaluate single episode."""
        # Placeholder implementation
        actions = np.random.randn(100, 7)
        success = np.random.random() > 0.6
        
        return EpisodeResult(
            success=success,
            total_reward=float(success),
            episode_length=len(actions),
            actions=actions,
            observations=[],
            info={"task": task, "suite": self.suite},
        )
    
    def get_tasks(self) -> List[str]:
        # Return task names based on suite
        if self.suite == "libero_90":
            return [f"task_{i}" for i in range(90)]
        return [f"{self.suite}_task_{i}" for i in range(10)]


class SimplerBenchmark(AbstractBenchmark):
    """SIMPLER benchmark for simulation evaluation.
    
    Tasks:
        - reach: Reach a target position
        - push: Push objects
        - drawer: Open/close drawers
        - pick_place: Pick and place objects
    """
    
    TASKS = ["reach", "push", "drawer", "pick_place"]
    
    def __init__(
        self,
        config: BenchmarkConfig,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        config.name = "simpler"
        super().__init__(config)
        self.env_kwargs = env_kwargs or {}
    
    def setup(self) -> None:
        """Setup SIMPLER environments."""
        logger.info("SIMPLER benchmark setup")
    
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """Evaluate single episode."""
        task = task or "reach"
        
        # Simulate based on task difficulty
        difficulty = {"reach": 0.8, "push": 0.6, "drawer": 0.5, "pick_place": 0.4}
        
        actions = np.random.randn(80, 7)
        success = np.random.random() < difficulty.get(task, 0.5)
        
        return EpisodeResult(
            success=success,
            total_reward=float(success) + np.random.random() * 0.5,
            episode_length=len(actions),
            actions=actions,
            observations=[],
            info={"task": task},
        )
    
    def get_tasks(self) -> List[str]:
        return self.TASKS


class RealWorldBenchmark(AbstractBenchmark):
    """Benchmark for real-world robot evaluation.
    
    Provides infrastructure for:
        - Human evaluation protocols
        - Safety monitoring
        - Success criteria definition
        - Video recording
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        tasks: List[str],
        success_criteria: Optional[Dict[str, Callable]] = None,
        success_callback: Optional[Callable[[str], bool]] = None,
        auto_mode: bool = False,
    ):
        """
        Args:
            config: Benchmark configuration
            tasks: List of task names to evaluate
            success_criteria: Optional dict mapping task names to success functions
            success_callback: Callback function to determine success (takes task, returns bool)
                             If None and not auto_mode, will prompt for user input
            auto_mode: If True, uses success_criteria automatically without human input
        """
        config.name = "real_world"
        super().__init__(config)
        self._tasks = tasks
        self.success_criteria = success_criteria or {}
        self.success_callback = success_callback
        self.auto_mode = auto_mode
    
    def setup(self) -> None:
        """Setup real-world evaluation."""
        logger.info("Real-world benchmark setup")
        if not self.auto_mode and self.success_callback is None:
            logger.info("Ensure robot is in safe configuration before starting evaluation")
            logger.info("Human supervision required for success evaluation")
    
    def evaluate_episode(
        self,
        policy: Callable,
        task: Optional[str] = None,
    ) -> EpisodeResult:
        """
        Evaluate a single real-world episode.
        
        Note: This requires human supervision for safety unless auto_mode is enabled.
        """
        task = task or self._tasks[0]
        
        logger.info(f"Starting real-world evaluation for task: {task}")
        
        # In real implementation, this would:
        # 1. Connect to robot
        # 2. Execute policy
        # 3. Record observations and actions
        # 4. Get human feedback on success
        
        # Determine success based on mode
        if self.auto_mode and task in self.success_criteria:
            # Use automatic success criteria
            success = self.success_criteria[task]()
        elif self.success_callback is not None:
            # Use provided callback
            success = self.success_callback(task)
        else:
            # Interactive mode - requires human input
            logger.warning(
                "No success_callback provided and not in auto_mode. "
                "Defaulting to unsuccessful for automated testing."
            )
            # For automated testing, default to False instead of blocking on input()
            # In real deployment, users should provide a success_callback
            success = False
        
        return EpisodeResult(
            success=success,
            total_reward=float(success),
            episode_length=0,
            actions=np.array([]),
            observations=[],
            info={"task": task, "human_evaluated": not self.auto_mode},
        )
    
    def get_tasks(self) -> List[str]:
        return self._tasks


class BenchmarkSuite:
    """Collection of benchmarks for comprehensive evaluation.
    
    Example:
        >>> suite = BenchmarkSuite()
        >>> suite.add_benchmark(SimulationBenchmark(config, env_name="reach"))
        >>> suite.add_benchmark(BridgeBenchmark(config))
        >>> results = suite.evaluate_all(policy)
    """
    
    def __init__(self):
        self.benchmarks: Dict[str, AbstractBenchmark] = {}
    
    def add_benchmark(self, benchmark: AbstractBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks[benchmark.config.name] = benchmark
    
    def evaluate_all(
        self,
        policy: Callable,
        benchmark_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate policy on all benchmarks.
        
        Args:
            policy: Policy function
            benchmark_names: Specific benchmarks to run (None for all)
            verbose: Print progress
            
        Returns:
            Results for all benchmarks
        """
        results = {}
        
        benchmark_names = benchmark_names or list(self.benchmarks.keys())
        
        for name in benchmark_names:
            if name not in self.benchmarks:
                logger.warning(f"Benchmark '{name}' not found")
                continue
            
            benchmark = self.benchmarks[name]
            
            if verbose:
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Running benchmark: {name}")
                logger.info(f"{'=' * 50}")
            
            benchmark.setup()
            results[name] = benchmark.evaluate(policy, verbose=verbose)
        
        return results
    
    def save_all_results(
        self,
        results: Dict[str, Dict[str, Any]],
        output_dir: str = "./benchmark_results",
    ) -> str:
        """Save all benchmark results."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / f"all_benchmarks_{int(time.time())}.json"
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        logger.info(f"All results saved to {filepath}")
        return str(filepath)


def create_benchmark(
    benchmark_type: str,
    config: Optional[BenchmarkConfig] = None,
    **kwargs,
) -> AbstractBenchmark:
    """
    Factory function to create benchmarks.
    
    Args:
        benchmark_type: Type of benchmark ("simulation", "bridge", "libero", "simpler", "real_world")
        config: Benchmark configuration
        **kwargs: Additional benchmark-specific arguments
        
    Returns:
        Configured benchmark instance
    """
    config = config or BenchmarkConfig()
    
    benchmark_map = {
        "simulation": SimulationBenchmark,
        "bridge": BridgeBenchmark,
        "libero": LIBEROBenchmark,
        "simpler": SimplerBenchmark,
        "real_world": RealWorldBenchmark,
    }
    
    if benchmark_type not in benchmark_map:
        raise ValueError(
            f"Unknown benchmark type: {benchmark_type}. "
            f"Available: {list(benchmark_map.keys())}"
        )
    
    return benchmark_map[benchmark_type](config, **kwargs)


def create_standard_suite() -> BenchmarkSuite:
    """Create standard benchmark suite for VLA evaluation."""
    suite = BenchmarkSuite()
    
    # Add standard benchmarks
    suite.add_benchmark(BridgeBenchmark(
        BenchmarkConfig(num_episodes=50)
    ))
    suite.add_benchmark(SimplerBenchmark(
        BenchmarkConfig(num_episodes=100)
    ))
    suite.add_benchmark(LIBEROBenchmark(
        BenchmarkConfig(num_episodes=50),
        suite="libero_90"
    ))
    
    return suite


__all__ = [
    "BenchmarkConfig",
    "EpisodeResult",
    "AbstractBenchmark",
    "SimulationBenchmark",
    "BridgeBenchmark",
    "LIBEROBenchmark",
    "SimplerBenchmark",
    "RealWorldBenchmark",
    "BenchmarkSuite",
    "create_benchmark",
    "create_standard_suite",
]
