"""
Experiment Runner Module

Single source of truth for:
- MLflow experiment setup
- DAGsHub integration
- Run lifecycle management

Trainer is intentionally MLflow-agnostic.
"""

import yaml
from pathlib import Path
import mlflow
import dagshub

from models import LeNetVariant
from trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunner:
    """
    Orchestrates controlled architecture experiments.

    Responsibilities:
    - MLflow + DAGsHub initialization
    - Experiment ownership
    - Run-level metadata & artifacts
    - Sequential model evaluation
    """

    def __init__(self, project_root):
        self.root = Path(project_root).resolve()

        # -----------------------------
        # Load configs
        # -----------------------------
        with open(self.root / "configs" / "config.yaml") as f:
            self.base_cfg = yaml.safe_load(f)

        with open(self.root / "configs" / "models.yaml") as f:
            self.model_cfg_paths = yaml.safe_load(f)["models"]

        # -----------------------------
        # MLflow + DAGsHub (ONE TIME)
        # -----------------------------
        dagshub.init(
            repo_owner="Y-R-A-V-R-5",
            repo_name="DownScaleXR",
            mlflow=True
        )

        mlflow.set_tracking_uri(
            self.base_cfg["mlflow"]["tracking_uri"]
        )

        mlflow.set_experiment(
            self.base_cfg["mlflow"]["experiment_name"]
        )

        mlflow.autolog()

        print(
            f"📂 ExperimentRunner ready | "
            f"{len(self.model_cfg_paths)} models queued"
        )

    def run_all(self):
        """Run all model experiments sequentially."""

        for model_cfg_relpath in self.model_cfg_paths:
            model_cfg_path = self.root / model_cfg_relpath

            with open(model_cfg_path) as f:
                model_cfg = yaml.safe_load(f)

            cfg = dict(self.base_cfg)
            cfg["model_name"] = model_cfg["model_name"]

            run_name = cfg["model_name"]
            print(f"\n🚀 Running: {run_name}")

            with mlflow.start_run(run_name=run_name):

                # -----------------------------
                # Tags (high-level semantics)
                # -----------------------------
                mlflow.set_tags({
                    "project": cfg["project"]["name"],
                    "model_name": run_name,
                    "downsampling": run_name.replace("lenet_", ""),
                    "device": "cpu",
                    "phase": "architecture_ablation",
                    "compression": "disabled"
                })

                # -----------------------------
                # Artifacts & params
                # -----------------------------
                mlflow.log_artifact(str(model_cfg_path))
                mlflow.log_artifact(
                    str(self.root / "configs" / "config.yaml")
                )

                mlflow.log_params(cfg["training"])

                # -----------------------------
                # Build & train
                # -----------------------------
                model = LeNetVariant(model_cfg_path)

                trainer = Trainer(
                    model=model,
                    config=cfg,
                    root_dir=self.root
                )

                trainer.run()


def run_experiments():
    runner = ExperimentRunner(PROJECT_ROOT)
    runner.run_all()
    return runner