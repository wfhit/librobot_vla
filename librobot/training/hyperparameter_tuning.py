"""Hyperparameter tuning utilities using Ray Tune and Optuna."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from librobot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchSpace:
    """Define hyperparameter search space.

    Example:
        >>> space = SearchSpace()
        >>> space.add_uniform("lr", 1e-5, 1e-3)
        >>> space.add_choice("batch_size", [16, 32, 64])
        >>> space.add_loguniform("weight_decay", 1e-6, 1e-2)
    """

    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_uniform(self, name: str, low: float, high: float) -> "SearchSpace":
        """Add uniform distribution parameter."""
        self.params[name] = {"type": "uniform", "low": low, "high": high}
        return self

    def add_loguniform(self, name: str, low: float, high: float) -> "SearchSpace":
        """Add log-uniform distribution parameter."""
        self.params[name] = {"type": "loguniform", "low": low, "high": high}
        return self

    def add_choice(self, name: str, choices: List[Any]) -> "SearchSpace":
        """Add categorical choice parameter."""
        self.params[name] = {"type": "choice", "choices": choices}
        return self

    def add_int(self, name: str, low: int, high: int) -> "SearchSpace":
        """Add integer range parameter."""
        self.params[name] = {"type": "int", "low": low, "high": high}
        return self

    def add_quniform(self, name: str, low: float, high: float, q: float) -> "SearchSpace":
        """Add quantized uniform distribution parameter."""
        self.params[name] = {"type": "quniform", "low": low, "high": high, "q": q}
        return self

    def to_ray(self) -> Dict[str, Any]:
        """Convert to Ray Tune search space."""
        try:
            from ray import tune

            ray_space = {}
            for name, config in self.params.items():
                param_type = config["type"]

                if param_type == "uniform":
                    ray_space[name] = tune.uniform(config["low"], config["high"])
                elif param_type == "loguniform":
                    ray_space[name] = tune.loguniform(config["low"], config["high"])
                elif param_type == "choice":
                    ray_space[name] = tune.choice(config["choices"])
                elif param_type == "int":
                    ray_space[name] = tune.randint(config["low"], config["high"] + 1)
                elif param_type == "quniform":
                    ray_space[name] = tune.quniform(config["low"], config["high"], config["q"])

            return ray_space

        except ImportError:
            raise ImportError("ray[tune] required for Ray Tune. Install with: pip install ray[tune]")

    def to_optuna(self, trial) -> Dict[str, Any]:
        """Convert to Optuna trial suggestions."""
        optuna_params = {}

        for name, config in self.params.items():
            param_type = config["type"]

            if param_type == "uniform":
                optuna_params[name] = trial.suggest_float(name, config["low"], config["high"])
            elif param_type == "loguniform":
                optuna_params[name] = trial.suggest_float(name, config["low"], config["high"], log=True)
            elif param_type == "choice":
                optuna_params[name] = trial.suggest_categorical(name, config["choices"])
            elif param_type == "int":
                optuna_params[name] = trial.suggest_int(name, config["low"], config["high"])
            elif param_type == "quniform":
                optuna_params[name] = trial.suggest_float(
                    name, config["low"], config["high"], step=config["q"]
                )

        return optuna_params


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Args:
        num_trials: Number of trials to run
        max_concurrent_trials: Maximum concurrent trials
        metric: Metric to optimize
        mode: Optimization mode ("min" or "max")
        search_algorithm: Search algorithm ("random", "grid", "bayesian", "hyperband")
        scheduler: Trial scheduler ("asha", "pbt", "median", None)
        grace_period: Minimum iterations before stopping
        reduction_factor: Factor for successive halving
        time_budget_s: Total time budget in seconds
        resources_per_trial: Resources per trial (CPU, GPU)
        local_dir: Local directory for results
    """
    num_trials: int = 10
    max_concurrent_trials: int = 4
    metric: str = "val_loss"
    mode: str = "min"
    search_algorithm: str = "bayesian"
    scheduler: Optional[str] = "asha"
    grace_period: int = 1
    reduction_factor: int = 2
    time_budget_s: Optional[int] = None
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {"cpu": 1, "gpu": 0.5})
    local_dir: str = "./ray_results"


class AbstractTuner(ABC):
    """Abstract base class for hyperparameter tuners."""

    @abstractmethod
    def tune(
        self,
        train_fn: Callable,
        search_space: SearchSpace,
        config: TuningConfig,
    ) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        pass

    @abstractmethod
    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        pass

    @abstractmethod
    def get_results_df(self):
        """Get results as DataFrame."""
        pass


class RayTuner(AbstractTuner):
    """Ray Tune hyperparameter tuner.

    Features:
        - Parallel trial execution
        - Multiple search algorithms (Bayesian, HyperBand, etc.)
        - Early stopping schedulers
        - Integration with experiment tracking
        - Distributed execution

    Example:
        >>> def train_fn(config):
        ...     model = create_model(lr=config["lr"])
        ...     for epoch in range(10):
        ...         loss = train_epoch(model)
        ...         tune.report(loss=loss)

        >>> search_space = SearchSpace().add_loguniform("lr", 1e-5, 1e-3)
        >>> tuner = RayTuner()
        >>> results = tuner.tune(train_fn, search_space, TuningConfig())
    """

    def __init__(self):
        self._results = None
        self._best_config = None

    def tune(
        self,
        train_fn: Callable,
        search_space: SearchSpace,
        config: TuningConfig,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning with Ray Tune.

        Args:
            train_fn: Training function that takes config dict
            search_space: Hyperparameter search space
            config: Tuning configuration

        Returns:
            Best hyperparameter configuration
        """
        try:
            from ray import tune
            from ray.tune import CLIReporter
            from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, MedianStoppingRule
            from ray.tune.search.optuna import OptunaSearch
            from ray.tune.search.hyperopt import HyperOptSearch
            from ray.tune.search import BasicVariantGenerator

            # Create search algorithm
            if config.search_algorithm == "bayesian":
                search_alg = OptunaSearch(
                    metric=config.metric,
                    mode=config.mode,
                )
            elif config.search_algorithm == "hyperopt":
                search_alg = HyperOptSearch(
                    metric=config.metric,
                    mode=config.mode,
                )
            else:
                search_alg = BasicVariantGenerator()

            # Create scheduler
            scheduler = None
            if config.scheduler == "asha":
                scheduler = ASHAScheduler(
                    metric=config.metric,
                    mode=config.mode,
                    grace_period=config.grace_period,
                    reduction_factor=config.reduction_factor,
                )
            elif config.scheduler == "pbt":
                scheduler = PopulationBasedTraining(
                    metric=config.metric,
                    mode=config.mode,
                    perturbation_interval=1,
                )
            elif config.scheduler == "median":
                scheduler = MedianStoppingRule(
                    metric=config.metric,
                    mode=config.mode,
                    grace_period=config.grace_period,
                )

            # Create reporter
            reporter = CLIReporter(
                metric_columns=[config.metric],
                parameter_columns=list(search_space.params.keys())[:3],
            )

            # Run tuning
            self._results = tune.run(
                train_fn,
                config=search_space.to_ray(),
                num_samples=config.num_trials,
                max_concurrent_trials=config.max_concurrent_trials,
                search_alg=search_alg,
                scheduler=scheduler,
                progress_reporter=reporter,
                resources_per_trial=config.resources_per_trial,
                local_dir=config.local_dir,
                time_budget_s=config.time_budget_s,
                verbose=1,
            )

            # Get best config
            self._best_config = self._results.get_best_config(
                metric=config.metric,
                mode=config.mode,
            )

            best_trial = self._results.get_best_trial(
                metric=config.metric,
                mode=config.mode,
            )

            logger.info(
                f"Best trial: {best_trial.last_result[config.metric]:.4f} "
                f"with config: {self._best_config}"
            )

            return self._best_config

        except ImportError as e:
            raise ImportError(
                f"ray[tune] required for Ray Tune. Install with: pip install 'ray[tune]': {e}"
            )

    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        if self._best_config is None:
            raise RuntimeError("No tuning results available. Run tune() first.")
        return self._best_config

    def get_results_df(self):
        """Get results as DataFrame."""
        if self._results is None:
            raise RuntimeError("No tuning results available. Run tune() first.")
        return self._results.dataframe()


class OptunaTuner(AbstractTuner):
    """Optuna hyperparameter tuner.

    Features:
        - Efficient Bayesian optimization
        - Pruning of unpromising trials
        - Multi-objective optimization
        - Visualization tools
        - Lightweight and easy to use

    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        ...     model = create_model(lr=lr)
        ...     loss = train_and_evaluate(model)
        ...     return loss

        >>> tuner = OptunaTuner()
        >>> best_params = tuner.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ):
        self._study = None
        self._study_name = study_name
        self._storage = storage
        self._load_if_exists = load_if_exists

    def tune(
        self,
        train_fn: Callable,
        search_space: SearchSpace,
        config: TuningConfig,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning with Optuna.

        Args:
            train_fn: Training function that takes config dict and returns metric
            search_space: Hyperparameter search space
            config: Tuning configuration

        Returns:
            Best hyperparameter configuration
        """
        try:
            import optuna

            # Create objective function that wraps train_fn
            def objective(trial):
                # Convert search space to Optuna suggestions
                params = search_space.to_optuna(trial)

                # Run training
                result = train_fn(params)

                # Return metric value
                if isinstance(result, dict):
                    return result.get(config.metric, result)
                return result

            # Create pruner
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=config.grace_period,
            )

            # Create sampler
            if config.search_algorithm == "bayesian":
                sampler = optuna.samplers.TPESampler()
            elif config.search_algorithm == "random":
                sampler = optuna.samplers.RandomSampler()
            elif config.search_algorithm == "grid":
                # Grid search requires discrete values - only include choice parameters
                grid_params = {}
                for name, spec in search_space.params.items():
                    if spec.get("type") == "choice" and "choices" in spec:
                        grid_params[name] = spec["choices"]
                    else:
                        logger.warning(
                            f"Parameter '{name}' is not a choice type, skipping for grid search"
                        )
                if not grid_params:
                    raise ValueError(
                        "Grid search requires at least one categorical (choice) parameter"
                    )
                sampler = optuna.samplers.GridSampler(grid_params)
            else:
                sampler = optuna.samplers.TPESampler()

            # Create or load study
            direction = "minimize" if config.mode == "min" else "maximize"
            self._study = optuna.create_study(
                study_name=self._study_name,
                storage=self._storage,
                load_if_exists=self._load_if_exists,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )

            # Run optimization
            self._study.optimize(
                objective,
                n_trials=config.num_trials,
                timeout=config.time_budget_s,
                n_jobs=config.max_concurrent_trials,
                show_progress_bar=True,
            )

            logger.info(
                f"Best trial: {self._study.best_value:.4f} "
                f"with config: {self._study.best_params}"
            )

            return self._study.best_params

        except ImportError as e:
            raise ImportError(f"optuna required. Install with: pip install optuna: {e}")

    def optimize(
        self,
        objective: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        direction: str = "minimize",
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct Optuna optimization interface.

        Args:
            objective: Objective function (takes trial, returns value)
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            direction: Optimization direction ("minimize" or "maximize")
            study_name: Study name for persistence

        Returns:
            Best hyperparameter configuration
        """
        try:
            import optuna

            self._study = optuna.create_study(
                study_name=study_name or self._study_name,
                storage=self._storage,
                load_if_exists=self._load_if_exists,
                direction=direction,
            )

            self._study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )

            return self._study.best_params

        except ImportError as e:
            raise ImportError(f"optuna required. Install with: pip install optuna: {e}")

    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        if self._study is None:
            raise RuntimeError("No tuning results available. Run tune() first.")
        return self._study.best_params

    def get_results_df(self):
        """Get results as DataFrame."""
        if self._study is None:
            raise RuntimeError("No tuning results available. Run tune() first.")
        return self._study.trials_dataframe()

    def visualize_optimization_history(self):
        """Visualize optimization history."""
        if self._study is None:
            raise RuntimeError("No tuning results available. Run tune() first.")

        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self._study)
        except ImportError:
            logger.warning("optuna visualization requires plotly")

    def visualize_param_importances(self):
        """Visualize parameter importances."""
        if self._study is None:
            raise RuntimeError("No tuning results available. Run tune() first.")

        try:
            import optuna.visualization as vis
            return vis.plot_param_importances(self._study)
        except ImportError:
            logger.warning("optuna visualization requires plotly")


def create_tuner(
    backend: str = "optuna",
    **kwargs,
) -> AbstractTuner:
    """
    Factory function to create hyperparameter tuner.

    Args:
        backend: Tuning backend ("ray", "optuna")
        **kwargs: Backend-specific arguments

    Returns:
        Configured hyperparameter tuner
    """
    if backend == "ray":
        return RayTuner()
    elif backend == "optuna":
        return OptunaTuner(**kwargs)
    else:
        raise ValueError(f"Unknown tuning backend: {backend}. Available: ray, optuna")


def get_vla_search_space() -> SearchSpace:
    """Get default search space for VLA training."""
    return (
        SearchSpace()
        .add_loguniform("learning_rate", 1e-6, 1e-3)
        .add_choice("batch_size", [8, 16, 32, 64])
        .add_loguniform("weight_decay", 1e-6, 1e-2)
        .add_uniform("warmup_ratio", 0.01, 0.1)
        .add_choice("optimizer", ["adamw", "adam", "sgd"])
        .add_choice("scheduler", ["cosine", "linear", "constant"])
        .add_uniform("gradient_clip", 0.5, 2.0)
        .add_choice("mixed_precision", [True, False])
    )


__all__ = [
    "SearchSpace",
    "TuningConfig",
    "AbstractTuner",
    "RayTuner",
    "OptunaTuner",
    "create_tuner",
    "get_vla_search_space",
]
