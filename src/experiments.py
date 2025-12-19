"""
Experiment Runner Module

This module orchestrates multiple ML experiments using:
- YAML-based configuration files
- MLflow for experiment tracking
- DagsHub as the remote MLflow backend

It loads a base training config, iterates over multiple model configs,
and runs each experiment end-to-end (model init → training → logging).
"""

import yaml
from pathlib import Path
import mlflow
import dagshub

from models import LeNetVariant
from trainer import Trainer


# Project root directory (two levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunner:
    """
    Manages and executes multiple experiments defined via config files.

    Responsibilities:
    - Load base and model-specific configurations
    - Initialize DagsHub + MLflow tracking
    - Run training for each model configuration
    - Log parameters, artifacts, and metadata to MLflow
    """

    def __init__(self, project_root):
        """
        Initialize the experiment runner.

        Args:
            project_root (str | Path): Root directory of the project
        """
        self.root = Path(project_root).resolve()

        # -----------------------------
        # Load base (shared) config
        # -----------------------------
        with open(self.root / "configs" / "config.yaml") as f:
            self.base_cfg = yaml.safe_load(f)

        # -----------------------------
        # Load list of model config paths
        # -----------------------------
        with open(self.root / "configs" / "models.yaml") as f:
            self.model_cfg_paths = yaml.safe_load(f)["models"]

        # -----------------------------
        # DagsHub + MLflow setup
        # -----------------------------
        dagshub.init(
            repo_owner="Y-R-A-V-R-5",
            repo_name="DownScaleXR",
            mlflow=True
        )

        # Automatically logs metrics, losses, and parameters where possible
        mlflow.autolog()

        print(f"📂 ExperimentRunner initialized with {len(self.model_cfg_paths)} models.")

    def run_all(self):
        """
        Run all experiments sequentially.

        For each model configuration:
        - Merge it with the base config
        - Start an MLflow run
        - Log configs, parameters, and metadata
        - Train the model using the Trainer class
        """
        for model_cfg_relpath in self.model_cfg_paths:
            model_cfg_path = self.root / model_cfg_relpath

            # Load model-specific configuration
            with open(model_cfg_path) as f:
                model_cfg = yaml.safe_load(f)

            # Create a fresh copy of the base config
            # and inject the model name
            cfg = dict(self.base_cfg)
            cfg["model_name"] = model_cfg["model_name"]

            run_name = cfg["model_name"]
            print(f"\n🚀 Running experiment: {run_name}")

            # -----------------------------
            # Start MLflow experiment run
            # -----------------------------
            with mlflow.start_run(run_name=run_name):

                # High-level tags for filtering & comparison
                mlflow.set_tags({
                    "model": run_name,
                    "downsampling": run_name.replace("lenet_", ""),
                    "device": "cpu",
                    "project": cfg["project"]["name"]
                })

                # Log configuration files as artifacts
                mlflow.log_artifact(str(model_cfg_path))
                mlflow.log_artifact(str(self.root / "configs" / "config.yaml"))

                # Log training hyperparameters
                mlflow.log_params(cfg["training"])

                # -----------------------------
                # Model initialization
                # -----------------------------
                model = LeNetVariant(model_cfg_path)

                # -----------------------------
                # Trainer initialization
                # -----------------------------
                trainer = Trainer(
                    model=model,
                    config=cfg,
                    root_dir=self.root
                )

                # -----------------------------
                # Run training loop
                # -----------------------------
                trainer.run()


def run_experiments():
    """
    Convenience entry point for scripts or notebooks.

    Creates an ExperimentRunner and executes all experiments.
    """
    runner = ExperimentRunner(PROJECT_ROOT)
    runner.run_all()
    return runner

# ============================================================
# End of experiments.py
# ============================================================