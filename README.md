# Chest X-ray Pneumonia Classification with Centralized + Federated Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **中文文档 / Chinese version:** [README_ZH.md](README_ZH.md)

A complete PyTorch experiment that compares **centralized deep learning** with **federated learning (FedAvg)** for binary pneumonia classification on chest X-ray images. The pipeline covers data loading, non-IID client simulation, model training, full evaluation, visualization, and Grad-CAM explainability.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Results](#results)
- [Explainability (Grad-CAM)](#explainability-grad-cam)
- [Reproducibility](#reproducibility)
- [File Reference](#file-reference)
- [License](#license)

---

## Overview

Pneumonia is a leading cause of mortality worldwide, and chest X-ray imaging is a primary diagnostic tool. In real healthcare settings, patient data is distributed across hospitals and **cannot be centralized** due to privacy regulations. This project explores two paradigms side by side:

1. **Centralized training** — all data is pooled in one place (the classical baseline).
2. **Federated training (FedAvg)** — data stays on simulated non-IID hospital clients; only model weights are aggregated by a central server.

Both approaches are trained on the same architecture (`ResNet-18` by default) and evaluated on the same held-out test set, so the comparison is apples-to-apples.

---

## Features

- **Two training paradigms** — centralized baseline + FedAvg federated learning
- **Non-IID client simulation** — configurable class-ratio skew across clients, with bounded, sample-conservative splits that avoid single-class collapse
- **Class-imbalance handling** — `BCEWithLogitsLoss(pos_weight=...)` + optional `WeightedRandomSampler` for centralized training
- **Full evaluation suite** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- **Rich visualizations** — loss/accuracy curves, confusion matrices, ROC curves, metric bar charts, sample prediction panels
- **Explainability** — Grad-CAM heatmaps highlighting diagnostic regions
- **Reproducible** — global seed for Python / NumPy / PyTorch + cuDNN deterministic mode
- **Single-file config** — everything tunable from `configs/config.yaml`

---

## Pipeline

```mermaid
flowchart TD
    subgraph Data["Data Layer"]
        D[Chest X-ray Dataset<br/>train / val / test]
    end

    subgraph Centralized["Centralized Path"]
        C1[WeightedRandomSampler]
        C2[ResNet-18 / SimpleCNN]
        C3[Train with BCEWithLogitsLoss]
        C4[Best model by val loss]
    end

    subgraph Federated["Federated Path (FedAvg)"]
        F1[Non-IID client split<br/>N clients, bounded skew]
        F2[Global model broadcast]
        F3[Local training per client]
        F4[Sample-weighted FedAvg aggregation]
        F5[Best global model by val loss]
    end

    D --> C1 --> C2 --> C3 --> C4
    D --> F1 --> F2 --> F3 --> F4 --> F5

    C4 --> E[Evaluate on test set<br/>metrics + plots + Grad-CAM]
    F5 --> E
```

---

## Project Structure

```text
.
├── configs/
│   └── config.yaml              # Single source of truth for all hyperparameters
├── src/
│   ├── data_loader.py           # Dataset loading, transforms, non-IID split
│   ├── model.py                 # SimpleCNN + ResNet-18 builders, Grad-CAM target layer
│   ├── train_centralized.py     # Centralized training entry point
│   ├── train_federated.py       # FedAvg federated training entry point
│   ├── client.py                # FederatedClient: local training logic
│   ├── server.py                # FedAvg weight aggregation
│   ├── evaluate.py              # Metrics, plots, Grad-CAM, standalone eval
│   └── utils.py                 # Seed, config, device, metrics helpers
├── outputs/                     # Generated artifacts (committed for reference)
│   ├── models/                  # Saved checkpoints (.pt)
│   ├── logs/                    # Training history + metrics (.json)
│   └── plots/                   # All figures (.png)
├── data/                        # Dataset (NOT tracked — provide your own)
├── .gitignore
├── LICENSE
├── MODIFICATION_LOG.md          # Development changelog
├── README.md                    # This file
├── README_ZH.md                 # Chinese README
└── requirements.txt
```

---

## Environment Setup

**Python:** 3.10 (recommended). Anaconda or venv both work.

```bash
# Option A — Anaconda
conda create -n fl_xray python=3.10
conda activate fl_xray
pip install -r requirements.txt

# Option B — venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `pyyaml`, `numpy`.

> **Note:** For GPU support, install the PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). CPU and Apple Silicon (MPS) are auto-detected and supported.

---

## Dataset

This repository does **not** include the dataset. We use the publicly available chest X-ray pneumonia dataset from Kermany et al. (2018):

- **Source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Reference:** Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.

Download and organize it in ImageFolder format:

```text
data/
├── train/
│   ├── NORMAL/      # 1,341 images
│   └── PNEUMONIA/   # 3,875 images
├── val/
│   ├── NORMAL/      # 8 images
│   └── PNEUMONIA/   # 8 images
└── test/
    ├── NORMAL/      # 234 images
    └── PNEUMONIA/   # 390 images
```

> The dataset is class-imbalanced (~3:1 PNEUMONIA:NORMAL in train), which this project explicitly addresses via `pos_weight` and weighted sampling.

---

## Quick Start

```bash
# 1. Train the centralized baseline (saves model + metrics + plots)
python src/train_centralized.py

# 2. Train the federated model (auto-detects centralized model for comparison)
python src/train_federated.py

# 3. (Optional) Re-evaluate saved checkpoints and regenerate all plots
python src/evaluate.py
```

All artifacts are written under `outputs/`.

---

## Configuration

Everything is controlled via [`configs/config.yaml`](configs/config.yaml):

| Section | Key Fields | Description |
|---------|-----------|-------------|
| `seed` | `42` | Global random seed |
| `data` | `image_size`, `num_workers`, `normalize_mean/std` | Input preprocessing |
| `model` | `name: resnet18` (`simple_cnn` also supported), `pretrained` | Model selection |
| `training` | `batch_size`, `epochs`, `learning_rate`, `weight_decay` | Centralized hyperparams |
| `federated` | `num_clients`, `rounds`, `local_epochs`, `client_lr`, `noniid_*` | FedAvg + non-IID controls |
| `paths` | `models_dir`, `plots_dir`, `logs_dir` | Output locations |

The non-IID split knobs (`noniid_ratio_span`, `noniid_min_ratio_floor`, `noniid_max_ratio_cap`, `noniid_min_samples_per_class`, `noniid_shuffle_target_ratios`) bound client class-ratio skew while guaranteeing exact sample conservation across clients.

---

## Methodology

### Centralized Training

- Loads the full `train/val/test` splits with train-time augmentation (random flip + rotation)
- Uses `WeightedRandomSampler` to balance mini-batches despite class imbalance
- Trains with `BCEWithLogitsLoss(pos_weight=num_neg/num_pos)` and Adam
- `ReduceLROnPlateau` scheduler (factor 0.5, patience 2) stabilizes optimization
- Saves the checkpoint with the lowest validation loss

### Federated Training (FedAvg)

- Splits `train` into `N` clients with **bounded non-IID label skew** (each client sees a different PNEUMONIA:NORMAL ratio, but no client collapses to a single class)
- Each round: server broadcasts the global model → each client trains locally for `E` epochs → server aggregates via **sample-size-weighted FedAvg**
- A single global `pos_weight` (computed across all client data) is passed to every client's local loss to counter imbalance
- Saves the global checkpoint with the lowest validation loss
- If a centralized model exists, automatically generates side-by-side comparison plots

---

## Results

Both models were trained with `ResNet-18` (from scratch, `seed=42`) and evaluated on the 624-image test set.

### Test-set Metrics

| Metric | Centralized | Federated (FedAvg) |
|--------|:-----------:|:------------------:|
| Accuracy | **88.62%** | 82.69% |
| Precision | **85.84%** | 78.66% |
| Recall | 97.95% | **99.23%** |
| F1 Score | **91.50%** | 87.76% |
| ROC-AUC | **0.9600** | 0.9449 |

### Confusion Matrices

| | Centralized | Federated |
|---|:---:|:---:|
| **TN / FP** | 171 / 63 | 129 / 105 |
| **FN / TP** | 8 / 382 | 3 / 387 |

**Observations:**
- The centralized model achieves higher overall accuracy and precision.
- The federated model achieves **near-perfect recall (99.23%)** — it catches almost all pneumonia cases — at the cost of more false positives. This is a clinically favorable trade-off (missing pneumonia is worse than a follow-up confirmatory scan).
- Despite non-IID client distributions, FedAvg closes most of the gap to centralized training, validating federated learning as a viable privacy-preserving alternative.

### Training Curves

<p align="center">
  <img src="outputs/plots/loss_curve_comparison.png" width="46%" alt="Loss curves">
  <img src="outputs/plots/accuracy_curve_comparison.png" width="46%" alt="Accuracy curves">
</p>
<p align="center"><em>Validation loss (left) and accuracy (right) — Centralized (per epoch) vs Federated (per round).</em></p>

### Metric Comparison

<p align="center">
  <img src="outputs/plots/centralized_vs_federated_bar.png" width="55%" alt="Metric comparison bar chart">
</p>

### Confusion Matrices & ROC Curves

<p align="center">
  <img src="outputs/plots/centralized_confusion_matrix.png" width="32%">
  <img src="outputs/plots/federated_confusion_matrix.png" width="32%">
  <img src="outputs/plots/centralized_vs_federated_bar.png" width="0%" style="visibility:hidden">
</p>
<p align="center"><em>Centralized (left) and Federated (right) confusion matrices on the test set.</em></p>

<p align="center">
  <img src="outputs/plots/centralized_roc_curve.png" width="46%">
  <img src="outputs/plots/federated_roc_curve.png" width="46%">
</p>
<p align="center"><em>ROC curves — Centralized AUC = 0.9600 (left), Federated AUC = 0.9449 (right).</em></p>

### Sample Predictions

<p align="center">
  <img src="outputs/plots/centralized_example_predictions.png" width="80%" alt="Centralized sample predictions">
</p>
<p align="center"><em>Centralized model predictions with probabilities on test images.</em></p>

---

## Explainability (Grad-CAM)

Grad-CAM highlights the image regions most influential to the model's prediction, providing a sanity check that the model attends to lung opacities rather than spurious artifacts.

<p align="center">
  <img src="outputs/plots/centralized_gradcam_examples.png" width="80%" alt="Centralized Grad-CAM">
</p>
<p align="center"><em>Grad-CAM overlays for the centralized model (original left, heatmap right).</em></p>

---

## Reproducibility

- A global seed (`config.seed`) is applied to `random`, `numpy`, `torch`, and `torch.cuda`
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` are set
- With the same config + seed + hardware, run-to-run variance is minimized
- Committed histories (`outputs/logs/*.json`) let `python src/evaluate.py` regenerate plots without retraining; model checkpoints (`outputs/models/*.pt`) are written locally and are not tracked

---

## File Reference

| File | Role |
|------|------|
| `src/data_loader.py` | `ImageFolder` datasets, train/eval transforms, `WeightedRandomSampler`, non-IID client split with bounded skew |
| `src/model.py` | `SimpleCNN`, `ResNet-18` builder with single-logit head, Grad-CAM target-layer resolver |
| `src/train_centralized.py` | Centralized training loop, `pos_weight`, `ReduceLROnPlateau`, best-model tracking |
| `src/train_federated.py` | FedAvg orchestration, global `pos_weight`, centralized comparison |
| `src/client.py` | `FederatedClient` — local training from broadcast global weights |
| `src/server.py` | `fedavg` — sample-size-weighted parameter aggregation |
| `src/evaluate.py` | Metrics, confusion matrix, ROC, bar chart, training curves, Grad-CAM, sample predictions |
| `src/utils.py` | Seed, config loader, device detection, binary metrics, serialization helpers |
| `configs/config.yaml` | All hyperparameters in one place |
| `outputs/` | Committed metrics JSON and plots for reference; model checkpoints are generated locally and not tracked |

---

## License

This project is released under the [MIT License](LICENSE).

The chest X-ray dataset is subject to its own license terms (see [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) and is **not** redistributed in this repository.
