"""
trainer.py

This module defines the training and evaluation pipeline for the DownScaleXR project.
It provides a CPU-only PyTorch trainer for binary classification on chest X-ray images.

Key Features:
- Config-driven training and evaluation
- Binary classification using BCEWithLogitsLoss
- Performance metrics: accuracy, precision, recall, F1-score, AUC
- Efficiency metrics: inference time, throughput, model size, parameter count, FLOPs
- Early stopping based on validation AUC
- Best-model checkpointing
- MLflow experiment tracking

The Trainer is designed to fairly compare baseline and downscaled models
in terms of both predictive performance and computational efficiency.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import numpy as np
from tqdm import tqdm
from pathlib import Path

from data import NpyDataset


class Trainer:
    """
    Trainer class for binary classification using PyTorch.
    Handles training, validation, efficiency evaluation, and MLflow logging.
    """

    def __init__(self, model, config, root_dir):
        """
        Initialize trainer.

        Args:
            model (nn.Module): PyTorch model
            config (dict): Experiment configuration
            root_dir (str | Path): Project root directory
        """

        self.model = model
        self.cfg = config
        self.root = Path(root_dir).resolve()
        self.device = torch.device("cpu")  # CPU-only execution

        self.model_name = self.cfg["model_name"]

        # Training hyperparameters
        self.batch_size = int(self.cfg["training"]["batch_size"])
        self.lr = float(self.cfg["training"]["lr"])
        self.epochs = int(self.cfg["training"]["epochs"])
        self.weight_decay = float(
            self.cfg["training"].get("weight_decay", 0.0)
        )

        # --------------------------------------------------
        # Dataset loading
        # --------------------------------------------------
        cfg_path = self.cfg["data"]["processed_path"].lstrip("/").lstrip("\\")
        proc_path = self.root / cfg_path
        fallback_path = self.root / "data" / "processed"

        if not proc_path.exists() and fallback_path.exists():
            print(f"⚠️ Dataset path not found. Using {fallback_path}")
            proc_path = fallback_path

        train_ds = NpyDataset(proc_path, split="train")
        val_ds = NpyDataset(proc_path, split="val")

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # --------------------------------------------------
        # MLflow setup
        # --------------------------------------------------
        mlflow_cfg = self.cfg.get("mlflow", {})
        if mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

        mlflow.set_experiment(
            mlflow_cfg.get("experiment_name", "DownScaleXR")
        )

    # --------------------------------------------------
    # Utility: performance metrics
    # --------------------------------------------------
    def compute_performance_metrics(self, y_true, y_probs):
        """
        Compute classification performance metrics.

        Args:
            y_true (list): Ground truth labels
            y_probs (list): Predicted probabilities

        Returns:
            dict: Metric name -> value
        """

        y_bin = np.round(y_probs)
        metrics = {}

        for name in self.cfg["evaluation"]["performance_metrics"]:
            try:
                if name == "accuracy":
                    metrics[name] = accuracy_score(y_true, y_bin)
                elif name == "precision":
                    metrics[name] = precision_score(
                        y_true, y_bin, zero_division=0
                    )
                elif name == "recall":
                    metrics[name] = recall_score(
                        y_true, y_bin, zero_division=0
                    )
                elif name == "f1_score":
                    metrics[name] = f1_score(
                        y_true, y_bin, zero_division=0
                    )
                elif name == "auc":
                    metrics[name] = roc_auc_score(y_true, y_probs)
            except ValueError:
                metrics[name] = 0.5 if name == "auc" else 0.0

        return metrics

    # --------------------------------------------------
    # Training for one epoch
    # --------------------------------------------------
    def train_epoch(self, epoch):
        """
        Train model for one epoch.
        """

        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]"
        ):
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            # Forward + backward pass
            self.optimizer.zero_grad()
            logits = self.model(images).squeeze()

            if logits.ndim == 0:
                logits = logits.unsqueeze(0)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        epoch_loss = running_loss / len(self.train_loader)
        metrics = self.compute_performance_metrics(all_labels, all_preds)

        return epoch_loss, metrics

    # --------------------------------------------------
    # Validation + efficiency evaluation
    # --------------------------------------------------
    def validate(self, epoch):
        """
        Validate model and compute efficiency metrics.
        """

        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        start_time = time.perf_counter()
        num_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch}/{self.epochs} [Val]",
                leave=False
            ):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(images).squeeze()
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)

                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                num_samples += images.size(0)

        elapsed_time = time.perf_counter() - start_time

        perf_metrics = self.compute_performance_metrics(
            all_labels, all_preds
        )

        efficiency_metrics = {
            "inference_time_ms": (elapsed_time / len(self.val_loader)) * 1000,
            "throughput_fps": num_samples / elapsed_time,
            "model_parameters": sum(
                p.numel() for p in self.model.parameters()
                if p.requires_grad
            ),
            "model_size_mb": sum(
                p.numel() * p.element_size()
                for p in self.model.parameters()
            ) / (1024 ** 2)
        }

        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss, perf_metrics, efficiency_metrics

    # --------------------------------------------------
    # Main training loop
    # --------------------------------------------------
    def run(self):
        """
        Run full training loop with tqdm progress bars and
        per-epoch validation summaries.
        """

        best_auc = 0.0
        patience = int(self.cfg["training"]["early_stopping_patience"])
        patience_counter = 0

        mlflow.set_tags({
            "model_name": self.model_name,
            "device": "cpu",
            "project": self.cfg["project"]["name"]
        })

        for epoch in range(1, self.epochs + 1):

            # -------------------------
            # Training
            # -------------------------
            train_loss, train_metrics = self.train_epoch(epoch)

            # -------------------------
            # Validation
            # -------------------------
            val_loss, val_metrics, eff_metrics = self.validate(epoch)

            # -------------------------
            # MLflow logging
            # -------------------------
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()},
                step=epoch
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items()},
                step=epoch
            )
            mlflow.log_metrics(eff_metrics, step=epoch)

            # -------------------------
            # Best model tracking
            # -------------------------
            is_best = val_metrics.get("auc", 0.0) > best_auc
            if is_best:
                best_auc = val_metrics["auc"]
                patience_counter = 0

                model_dir = self.root / "model" / self.model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    model_dir / "best_model.pt"
                )
            else:
                patience_counter += 1

            # -------------------------
            # Console summary
            # -------------------------
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            metric_str = " | ".join(
                f"{k.upper()}={v:.4f}"
                for k, v in val_metrics.items()
            )
            print(f"Val Metrics: {metric_str}")

            print(
                f"Best AUC so far: {best_auc:.4f} "
                f"{'✅' if is_best else ''}"
            )
            print("-" * 60)

            # -------------------------
            # Early stopping
            # -------------------------
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

        print(f"\n🏆 Final Best Validation AUC: {best_auc:.4f}") 