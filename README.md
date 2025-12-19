# DownScaleXR: Efficient Chest X-Ray Classification with Downscaled CNNs

## Overview

DownScaleXR is a lightweight, interpretable convolutional neural network (CNN) framework designed to classify chest X-ray (CXR) images as **NORMAL** or **PNEUMONIA**. The project systematically evaluates the impact of different downsampling strategies—**AvgPool, MaxPool, and Strided Convolutions**—on model performance, computational efficiency, and decision bias.

The core idea is to balance **predictive performance** with **CPU-friendly efficiency**, making the models suitable for deployment in resource-constrained clinical environments.

---

## Project Structure

```
DownScaleXR/
├─ configs/ # YAML configs for project and models
├─ data/
│ ├─ CXR/ # Raw dataset (train/val/test)
│ └─ processed/ # Preprocessed numpy arrays
├─ model/ # Trained model checkpoints (*.pt)
├─ notebook/
│ ├─ comparision.ipynb # MLflow experiment analysis
│ └─ inference.ipynb # Inference and confusion matrices
├─ src/
│ ├─ data.py # Dataset loader & preprocessing
│ ├─ models.py # LeNet variants
│ ├─ trainer.py # Training & evaluation pipeline
│ └─ experiments.py # ExperimentRunner with MLflow & DagsHub
└─ README.md
```
