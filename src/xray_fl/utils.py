"""Shared helpers: reproducibility, device selection, IO, and metrics."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def set_seed(seed: int) -> None:
    """Set random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    """Create directory path if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Return the CUDA device when available, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict, path: str | Path) -> None:
    """Save dictionary as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move all tensors in a state dict to CPU for safe serialization/aggregation."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def compute_binary_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    """Compute binary classification metrics from true labels and positive-class probabilities."""
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob_np >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
    }


def epoch_binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch binary accuracy using sigmoid-thresholded logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean().item()
