"""Metric helpers."""

import pytest
import torch

from xray_fl.utils import compute_binary_metrics, epoch_binary_accuracy


def test_compute_binary_metrics_on_perfect_predictions():
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_binary_metrics_counts_false_negatives():
    metrics = compute_binary_metrics([0, 1], [0.9, 0.1])

    assert metrics["accuracy"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 1], [1, 0]]


def test_epoch_binary_accuracy_thresholds_at_half():
    logits = torch.tensor([10.0, -10.0, 0.1])
    labels = torch.tensor([1.0, 0.0, 0.0])

    assert epoch_binary_accuracy(logits, labels) == pytest.approx(2 / 3)
