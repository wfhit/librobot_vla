"""Experiment tracking integrations for W&B and MLflow."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from librobot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking.

    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment/run
        tags: List of tags for the experiment
        notes: Notes/description for the experiment
        config: Hyperparameter configuration dictionary
        log_dir: Local directory for logs
        offline: Run in offline mode (no network)
    """

    project_name: str = "librobot-vla"
    experiment_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    log_dir: str = "./logs"
    offline: bool = False


class AbstractExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def init(self) -> None:
        """Initialize the experiment tracker."""
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to the tracker."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact (file or directory)."""
        pass

    @abstractmethod
    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image."""
        pass

    @abstractmethod
    def log_table(
        self,
        key: str,
        data: Any,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Log tabular data."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish and close the experiment."""
        pass

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class WandbTracker(AbstractExperimentTracker):
    """Weights & Biases experiment tracker.

    Features:
        - Real-time metric logging and visualization
        - Hyperparameter tracking
        - Model artifact versioning
        - Image and table logging
        - Experiment comparison
        - Team collaboration

    Example:
        >>> config = ExperimentConfig(
        ...     project_name="robot-training",
        ...     experiment_name="groot-v1",
        ...     config={"lr": 1e-4, "batch_size": 32},
        ... )
        >>> with WandbTracker(config) as tracker:
        ...     tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        entity: Optional[str] = None,
        resume: Optional[str] = None,
        reinit: bool = True,
    ):
        super().__init__(config)
        self.entity = entity
        self.resume = resume
        self.reinit = reinit
        self._run = None

    def init(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            # Set offline mode if requested
            if self.config.offline:
                os.environ["WANDB_MODE"] = "offline"

            # Initialize run
            self._run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                entity=self.entity,
                config=self.config.config,
                tags=self.config.tags,
                notes=self.config.notes,
                dir=self.config.log_dir,
                resume=self.resume,
                reinit=self.reinit,
            )

            self._is_initialized = True
            logger.info(f"W&B initialized: {self._run.url}")

        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            raise

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to W&B."""
        if not self._is_initialized:
            return

        import wandb

        wandb.log(metrics, step=step, commit=commit)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to W&B config."""
        if not self._is_initialized:
            return

        import wandb

        wandb.config.update(params)

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact to W&B."""
        if not self._is_initialized:
            return

        import wandb

        artifact_path = Path(artifact_path)
        artifact_name = name or artifact_path.stem

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
        )

        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))

        self._run.log_artifact(artifact)
        logger.info(f"Artifact logged: {artifact_name}")

    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image to W&B."""
        if not self._is_initialized:
            return

        import wandb

        if isinstance(image, wandb.Image):
            wandb.log({key: image}, step=step)
        else:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_table(
        self,
        key: str,
        data: Any,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Log tabular data to W&B."""
        if not self._is_initialized:
            return

        import wandb

        if isinstance(data, wandb.Table):
            table = data
        elif columns is not None:
            table = wandb.Table(columns=columns, data=data)
        else:
            table = wandb.Table(dataframe=data)

        self._run.log({key: table})

    def log_model(
        self,
        model: Any,
        model_name: str = "model",
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Log a model artifact with versioning."""
        if not self._is_initialized:
            return

        import tempfile

        import torch
        import wandb

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            artifact = wandb.Artifact(
                name=model_name,
                type="model",
            )
            artifact.add_file(f.name, name="model.pt")
            self._run.log_artifact(artifact, aliases=aliases)

        logger.info(f"Model logged: {model_name}")

    def watch(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch model gradients and parameters."""
        if not self._is_initialized:
            return

        import wandb

        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish W&B run."""
        if self._is_initialized and self._run is not None:
            import wandb

            wandb.finish()
            self._is_initialized = False
            logger.info("W&B run finished")


class MLflowTracker(AbstractExperimentTracker):
    """MLflow experiment tracker.

    Features:
        - Metric and parameter logging
        - Model registry and versioning
        - Artifact storage
        - Experiment comparison
        - Model serving integration

    Example:
        >>> config = ExperimentConfig(
        ...     project_name="robot-training",
        ...     experiment_name="groot-v1",
        ...     config={"lr": 1e-4, "batch_size": 32},
        ... )
        >>> with MLflowTracker(config) as tracker:
        ...     tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        super().__init__(config)
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self._run = None

    def init(self) -> None:
        """Initialize MLflow run."""
        try:
            import mlflow

            # Set tracking URI
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Set registry URI
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)

            # Set experiment
            mlflow.set_experiment(self.config.project_name)

            # Start run
            self._run = mlflow.start_run(
                run_name=self.config.experiment_name,
                tags={tag: "1" for tag in self.config.tags},
                description=self.config.notes,
            )

            # Log initial config
            if self.config.config:
                mlflow.log_params(self._flatten_dict(self.config.config))

            self._is_initialized = True
            logger.info(f"MLflow initialized: run_id={self._run.info.run_id}")

        except ImportError:
            logger.warning("mlflow not installed. Install with: pip install mlflow")
            raise

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to MLflow."""
        if not self._is_initialized:
            return

        import mlflow

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        if not self._is_initialized:
            return

        import mlflow

        mlflow.log_params(self._flatten_dict(params))

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact to MLflow."""
        if not self._is_initialized:
            return

        import mlflow

        artifact_path = Path(artifact_path)

        if artifact_path.is_dir():
            mlflow.log_artifacts(str(artifact_path), artifact_path=name)
        else:
            mlflow.log_artifact(str(artifact_path), artifact_path=name)

        logger.info(f"Artifact logged: {artifact_path}")

    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image to MLflow."""
        if not self._is_initialized:
            return

        import tempfile

        import mlflow
        import numpy as np

        # Convert to numpy if needed
        if hasattr(image, "numpy"):
            image = image.numpy()

        # Save and log
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            try:
                from PIL import Image

                if isinstance(image, np.ndarray):
                    img = Image.fromarray(image.astype(np.uint8))
                else:
                    img = image
                img.save(f.name)
                mlflow.log_artifact(f.name, artifact_path=f"images/{key}")
            except ImportError:
                logger.warning("PIL not available for image logging")

    def log_table(
        self,
        key: str,
        data: Any,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Log tabular data to MLflow."""
        if not self._is_initialized:
            return

        import tempfile

        import mlflow

        # Convert to DataFrame if needed
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                df = data
            elif columns is not None:
                df = pd.DataFrame(data, columns=columns)
            else:
                df = pd.DataFrame(data)

            # Save and log as CSV
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, artifact_path=f"tables/{key}.csv")

        except ImportError:
            logger.warning("pandas not available for table logging")

    def log_model(
        self,
        model: Any,
        model_name: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a PyTorch model to MLflow with optional registration."""
        if not self._is_initialized:
            return

        try:
            import mlflow.pytorch

            mlflow.pytorch.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=registered_model_name,
            )

            logger.info(f"Model logged: {model_name}")

        except ImportError:
            logger.warning("mlflow.pytorch not available")

    def finish(self) -> None:
        """Finish MLflow run."""
        if self._is_initialized and self._run is not None:
            import mlflow

            mlflow.end_run()
            self._is_initialized = False
            logger.info("MLflow run finished")


class MultiTracker(AbstractExperimentTracker):
    """Multi-tracker that logs to multiple backends simultaneously.

    Example:
        >>> config = ExperimentConfig(project_name="robot-training")
        >>> tracker = MultiTracker(config, backends=["wandb", "mlflow"])
        >>> with tracker:
        ...     tracker.log_metrics({"loss": 0.5})
    """

    def __init__(
        self,
        config: ExperimentConfig,
        backends: list[str] = None,
        **backend_kwargs,
    ):
        super().__init__(config)
        self.backends = backends or ["wandb"]
        self._trackers: list[AbstractExperimentTracker] = []
        self._backend_kwargs = backend_kwargs

    def init(self) -> None:
        """Initialize all trackers."""
        for backend in self.backends:
            try:
                if backend == "wandb":
                    tracker = WandbTracker(
                        self.config,
                        **self._backend_kwargs.get("wandb", {}),
                    )
                elif backend == "mlflow":
                    tracker = MLflowTracker(
                        self.config,
                        **self._backend_kwargs.get("mlflow", {}),
                    )
                else:
                    logger.warning(f"Unknown tracking backend: {backend}")
                    continue

                tracker.init()
                self._trackers.append(tracker)

            except Exception as e:
                logger.warning(f"Failed to initialize {backend}: {e}")

        self._is_initialized = len(self._trackers) > 0

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to all trackers."""
        for tracker in self._trackers:
            tracker.log_metrics(metrics, step=step, commit=commit)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to all trackers."""
        for tracker in self._trackers:
            tracker.log_params(params)

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """Log artifact to all trackers."""
        for tracker in self._trackers:
            tracker.log_artifact(artifact_path, name=name, artifact_type=artifact_type)

    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log image to all trackers."""
        for tracker in self._trackers:
            tracker.log_image(key, image, step=step, caption=caption)

    def log_table(
        self,
        key: str,
        data: Any,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Log table to all trackers."""
        for tracker in self._trackers:
            tracker.log_table(key, data, columns=columns)

    def finish(self) -> None:
        """Finish all trackers."""
        for tracker in self._trackers:
            tracker.finish()
        self._trackers = []
        self._is_initialized = False


def create_tracker(
    backend: str = "wandb",
    project_name: str = "librobot-vla",
    experiment_name: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    **kwargs,
) -> AbstractExperimentTracker:
    """
    Factory function to create experiment tracker.

    Args:
        backend: Tracking backend ("wandb", "mlflow", "multi")
        project_name: Project name
        experiment_name: Experiment/run name
        config: Hyperparameter configuration
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured experiment tracker

    Example:
        >>> tracker = create_tracker(
        ...     backend="wandb",
        ...     project_name="robot-training",
        ...     config={"lr": 1e-4},
        ... )
        >>> tracker.init()
        >>> tracker.log_metrics({"loss": 0.5})
        >>> tracker.finish()
    """
    exp_config = ExperimentConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        config=config or {},
        **{k: v for k, v in kwargs.items() if k in ExperimentConfig.__dataclass_fields__},
    )

    tracker_kwargs = {
        k: v for k, v in kwargs.items() if k not in ExperimentConfig.__dataclass_fields__
    }

    if backend == "wandb":
        return WandbTracker(exp_config, **tracker_kwargs)
    elif backend == "mlflow":
        return MLflowTracker(exp_config, **tracker_kwargs)
    elif backend == "multi":
        backends = tracker_kwargs.pop("backends", ["wandb", "mlflow"])
        return MultiTracker(exp_config, backends=backends, **tracker_kwargs)
    else:
        raise ValueError(f"Unknown tracking backend: {backend}. Available: wandb, mlflow, multi")


__all__ = [
    "ExperimentConfig",
    "AbstractExperimentTracker",
    "WandbTracker",
    "MLflowTracker",
    "MultiTracker",
    "create_tracker",
]
