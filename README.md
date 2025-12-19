# DownScaleXR — Downsampling-Induced Bias Under CPU Constraints in Chest X-ray CNNs

## Overview

**DownScaleXR** is a controlled architectural study that analyzes how **early spatial downsampling choices** influence **generalization, decision bias, and CPU inference behavior** in chest X-ray (CXR) classification.

The project focuses on **binary classification (NORMAL vs PNEUMONIA)** using intentionally lightweight CNNs to isolate *inductive bias introduced by downsampling*, rather than masking it behind depth, residuals, or modern architectural optimizations.

This is a **constraint-driven R&D study**, not a leaderboard exercise.

---

## Motivation

This project was driven by three practical realities:

- **CPU-only deployment**  
  Many clinical and edge environments cannot rely on GPUs.

- **Noisy, limited data**  
  Medical datasets amplify architectural bias.

- **Architecture literacy gap**  
  Pooling and strided convolutions are often treated as interchangeable — **they are not**.

### Core Question

> **How does spatial compression itself shape decision boundaries under limited supervision and CPU constraints?**

---

## Architectural Scope & Design Choices

To keep the study interpretable and controlled:

- A **LeNet-style CNN** was used to minimize confounding factors.
- Modern architectures (**ResNet, MobileNet, EfficientNet**) were intentionally avoided.
- Skip connections, depth-wise convolutions, and compound scaling dilute the observable effect of early downsampling.

**This project isolates downsampling behavior — not representation capacity.**

---

## Downsampling Strategies Studied

All variants share **identical depth, width, and parameter count (~11M)**.

### AvgPool
- Smooths spatial activations  
- Acts as an implicit regularizer  
- Produces conservative decision boundaries  

### MaxPool
- Amplifies high-activation regions  
- Improves recall but increases false positives  
- Prone to pathology over-prediction  

### Strided Convolutions
- Learnable downsampling  
- Under limited data, collapses to MaxPool-like behavior  

---

## Goals

- Quantify **performance vs bias trade-offs**
- Measure **real CPU latency and throughput**
- Analyze **generalization gaps under noise**
- Ensure full experiment reproducibility via **MLflow + DagsHub**

---

## Project Structure

```
DownScaleXR/
├─ configs/ # YAML-driven experiment configuration
├─ data/ # Raw and preprocessed CXR data
├─ model/ # Best checkpoints per variant
├─ artifacts/ # Metrics, plots, and inference visualizations
├─ notebooks/ # MLflow analysis & comparison
├─ scripts/ # Entry points and preprocessing
├─ src/ # Core training, models, experiments
├─ requirements.txt
└─ README.md

```

---

## Experiments

Three LeNet variants were trained on the same chest X-ray dataset with identical hyperparameters to evaluate how different downsampling strategies affect performance and efficiency.

| Model Name      | Downsampling | Val AUC  | Val F1  | Val Precision | Val Recall | Val Accuracy | Train Accuracy | Inference Time (ms) | Throughput (FPS) | Parameters | Model Size (MB) |
|-----------------|--------------|----------|---------|---------------|------------|--------------|----------------|-------------------|-----------------|------------|----------------|
| lenet_strided   | strided      | 0.895    | 0.820   | 0.697         | 0.997      | 0.727        | 0.989          | 50.77             | 608.65          | 11.4 M     | 43.47          |
| lenet_avgpool   | avgpool      | 0.890    | 0.814   | 0.688         | 0.997      | 0.715        | 0.980          | 78.20             | 395.13          | 11.4 M     | 43.47          |
| lenet_maxpool   | maxpool      | 0.854    | 0.837   | 0.723         | 0.992      | 0.757        | 0.996          | 168.70            | 183.17          | 11.4 M     | 43.47          |

> **Observation:**  
All models have roughly the same parameter count and model size (~11M params, 43 MB). Differences arise primarily from **downsampling strategy**, impacting inference speed, throughput, and class-specific decision biases.
---

## Confusion Matrices

The test set predictions highlight differences in model behavior based on downsampling strategy:

| Model          | Confusion Matrix |
|----------------|----------------|
| lenet_avgpool  | ![lenet_avgpool](./artifacts/inference/lenet_avgpool_side_by_side.png) |
| lenet_maxpool  | ![lenet_maxpool](./artifacts/inference/lenet_maxpool_side_by_side.png) |
| lenet_strided  | ![lenet_strided](./artifacts/inference/lenet_strided_side_by_side.png) |

**Insights:**
- **AvgPool:** Balanced errors; moderate false positives and false negatives. Conservative decision boundaries.
- **MaxPool:** High recall for pneumonia but over-predicts pathology. Bias toward positive class.
- **Strided Conv:** Behavior similar to MaxPool; collapses to same decision bias on limited data.

---

## Performance vs Efficiency

The models were evaluated for both predictive performance and CPU efficiency:

- **Inference Time:**  
  - AvgPool: Moderate speed, good balance  
  - Strided Conv: Slightly slower than AvgPool  
  - MaxPool: Slowest due to pooling overhead  

- **Throughput (FPS):**  
  - Strided Conv: Highest throughput  
  - AvgPool: Moderate  
  - MaxPool: Lowest  

- **Trade-offs:**  
  - AvgPool achieves a good balance between speed and accuracy.  
  - MaxPool gives high F1 but at a computational cost.  
  - Strided Conv achieves higher throughput but can propagate decision biases similar to MaxPool.  

### Visual Analysis

Key comparative plots:

- **Accuracy vs Latency:**  
  ![Accuracy vs Latency](./artifacts/comparision/accuracy_vs_latency.png)  

- **Accuracy vs Model Size:**  
  ![Accuracy vs Model Size](./artifacts/comparision/accuracy_vs_model_size.png)  

- **Accuracy vs Throughput:**  
  ![Accuracy vs Throughput](./artifacts/comparision/accuracy_vs_throughput.png)  

- **CPU Efficiency Overview:**  
  ![CPU Efficiency](./artifacts/comparision/cpu_efficiency.png)  

- **Generalization Gap (Train vs Val):**  
  ![Generalization Gap](./artifacts/comparision/generalization_gap.png)  

- **Validation Performance Summary:**  
  ![Validation Performance](./artifacts/comparision/validation_performance.png)  

---

## Architectural Conclusions

From the experiments, several architectural insights emerged:

1. **Downsampling choice is crucial for bias control:**  
   - AvgPool → smoother features, conservative errors  
   - MaxPool / Strided → aggressive features, prone to over-predict pathology  

2. **Small datasets amplify pooling biases:**  
   - Limited supervision exaggerates the effect of downsampling, especially for minority classes  

3. **Strided convolution is not inherently superior to MaxPool:**  
   - Behaves similarly under CPU-limited conditions and small datasets  

4. **CPU-friendly models:**  
   - All three variants remain lightweight (~11M parameters, 43 MB) and deployable in resource-constrained environments  

### Inference Visualizations

- **Side-by-Side Comparisons (Confusion Matrix, ROC, P-R Curve):**  
  ![lenet_avgpool](./artifacts/inference/lenet_avgpool_side_by_side.png)  
  ![lenet_maxpool](./artifacts/inference/lenet_maxpool_side_by_side.png)  
  ![lenet_strided](./artifacts/inference/lenet_strided_side_by_side.png)  

- **Model Comparison (Accuracy & F1 Score):**  
  ![Model Comparison](./artifacts/inference/model_comparison.png)  

---

## MLflow Tracking

All experiments were logged to **DagsHub MLflow** to ensure reproducibility, allow easy comparison, and facilitate structured analysis.

- **Tracking URI:**  
  `https://dagshub.com/Y-R-A-V-R-5/DownScaleXR.mlflow`

- **Experiment Name:**  
  `DownScaleXR`

- **Purpose:**  
  - Store all metrics (train/validation), parameters, and tags  
  - Log artifacts such as plots and model checkpoints  
  - Enable side-by-side comparisons of different downsampling strategies  

- **Typical Usage:**

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Y-R-A-V-R-5/DownScaleXR.mlflow")

# Select experiment
mlflow.set_experiment("DownScaleXR")

# Search for past runs
experiment = mlflow.get_experiment_by_name("DownScaleXR")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Display run metrics and parameters
print(runs.head())
```

### Logged Artifacts

- **Model checkpoints:** `model/<variant>/best_model.pt`  
- **Plots:** `artifacts/comparision/*.png` and `artifacts/inference/*.png`  
- **Configuration files:** `configs/*.yaml`  

### Logged Metrics

- **Performance:** `accuracy`, `precision`, `recall`, `f1_score`, `auc`  
- **Efficiency:** `inference_time_ms`, `throughput_fps`, `model_parameters`, `model_size_mb`  
- **Tracking:** Metrics logged per epoch for both training and validation  

> Using **MLflow + DagsHub** ensures reproducibility, enables easy experiment comparisons, and provides structured logging of both performance and efficiency metrics.

---








