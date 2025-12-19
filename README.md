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

---


---

## Experiments

Three LeNet variants were trained on the same dataset with identical hyperparameters:

| Model Name      | Downsampling | Val AUC  | Val F1  | Val Precision | Val Recall | Val Accuracy | Train Accuracy | Inference Time (ms) | Throughput (FPS) | Parameters | Model Size (MB) |
|-----------------|--------------|----------|---------|---------------|------------|--------------|----------------|-------------------|-----------------|------------|----------------|
| lenet_strided   | strided      | 0.895    | 0.820   | 0.697         | 0.997      | 0.727        | 0.989          | 50.77             | 608.65          | 11.4 M     | 43.47          |
| lenet_avgpool   | avgpool      | 0.890    | 0.814   | 0.688         | 0.997      | 0.715        | 0.980          | 78.20             | 395.13          | 11.4 M     | 43.47          |
| lenet_maxpool   | maxpool      | 0.854    | 0.837   | 0.723         | 0.992      | 0.757        | 0.996          | 168.70            | 183.17          | 11.4 M     | 43.47          |

> **Observation:**  
All models have roughly the same parameter count and model size (~11M params, 43 MB). Differences arise primarily from **downsampling strategy**, affecting inference speed, throughput, and class-specific biases.

---

## Confusion Matrices

The test set predictions reveal important behavior differences:

| Model          | Confusion Matrix |
|----------------|----------------|
| lenet_avgpool  | ![lenet_avgpool](./notebook/figures/lenet_avgpool_cm.png) |
| lenet_maxpool  | ![lenet_maxpool](./notebook/figures/lenet_maxpool_cm.png) |
| lenet_strided  | ![lenet_strided](./notebook/figures/lenet_strided_cm.png) |

**Insights:**
- **AvgPool:** Balanced errors; moderate false positives and false negatives. Conservative decision boundaries.
- **MaxPool:** High recall for pneumonia but over-predicts pathology. Bias toward positive class.
- **Strided Conv:** Behavior mirrors MaxPool; collapses to the same decision bias on limited data.

---

## Performance vs Efficiency

- **Inference Time:** AvgPool fastest on CPU due to less aggressive feature aggregation; Strided slightly slower; MaxPool slowest.
- **Throughput:** Strided highest, MaxPool lowest, reflecting pooling overhead.
- **Trade-off:** AvgPool balances speed and performance; MaxPool achieves high F1 but at computational cost.

---

## Architectural Conclusions

1. **Downsampling choice is crucial for bias control:**
   - AvgPool → smoother features, conservative errors
   - MaxPool / Strided → aggressive features, prone to over-predict pathology
2. **Small datasets amplify pooling biases.**
3. **Strided convolution is not inherently superior to MaxPool**; behaves similarly on limited supervision.
4. **CPU-friendly models:** All three variants remain lightweight and deployable in resource-constrained settings.

---

## MLflow Tracking

All experiments are logged to **DagsHub MLflow** for reproducibility and analysis:

```python
mlflow.set_tracking_uri("https://dagshub.com/Y-R-A-V-R-5/DownScaleXR.mlflow")
experiment_name = "DownScaleXR"
