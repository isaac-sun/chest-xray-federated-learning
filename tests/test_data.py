"""Invariants of the simulated non-IID client split."""

from types import SimpleNamespace

import numpy as np

from xray_fl.data import _compute_client_class1_quotas, split_noniid_indices

SPLIT_DEFAULTS = {
    "ratio_span": 0.10,
    "min_ratio_floor": 0.20,
    "max_ratio_cap": 0.85,
    "min_samples_per_class": 1,
    "shuffle_target_ratios": True,
}


def make_dataset(n_normal: int = 120, n_pneumonia: int = 360, seed: int = 0) -> SimpleNamespace:
    """Minimal stand-in for an ImageFolder dataset (only `.targets` is used)."""
    rng = np.random.default_rng(seed)
    targets = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_pneumonia, dtype=int)])
    rng.shuffle(targets)
    return SimpleNamespace(targets=targets.tolist())


def test_quotas_conserve_the_global_class1_count():
    client_sizes = np.array([100, 100, 100, 100, 100])

    quotas = _compute_client_class1_quotas(
        client_sizes,
        total_class1=240,
        min_ratio=0.30,
        max_ratio=0.70,
        min_samples_per_class=1,
    )

    assert quotas.sum() == 240
    assert np.all(quotas >= 0)
    assert np.all(quotas <= client_sizes)


def test_split_is_disjoint_and_covers_every_sample():
    dataset = make_dataset()

    clients = split_noniid_indices(dataset, num_clients=5, seed=42, **SPLIT_DEFAULTS)

    flat = np.concatenate([np.asarray(indices) for indices in clients])
    assert len(flat) == len(dataset.targets)
    assert len(np.unique(flat)) == len(dataset.targets)
    assert sorted(flat.tolist()) == list(range(len(dataset.targets)))


def test_split_keeps_clients_two_class_and_within_the_skew_band():
    dataset = make_dataset()
    targets = np.asarray(dataset.targets)
    global_ratio = targets.mean()
    lower = max(SPLIT_DEFAULTS["min_ratio_floor"], global_ratio - SPLIT_DEFAULTS["ratio_span"])
    upper = min(SPLIT_DEFAULTS["max_ratio_cap"], global_ratio + SPLIT_DEFAULTS["ratio_span"])

    clients = split_noniid_indices(dataset, num_clients=5, seed=42, **SPLIT_DEFAULTS)

    tolerance = 2.0 / min(len(indices) for indices in clients)
    for indices in clients:
        labels = targets[np.asarray(indices)]
        assert (labels == 0).sum() >= 1, "client collapsed to a single class"
        assert (labels == 1).sum() >= 1, "client collapsed to a single class"
        assert lower - tolerance <= labels.mean() <= upper + tolerance


def test_split_is_deterministic_for_a_seed():
    dataset = make_dataset()

    first = split_noniid_indices(dataset, num_clients=5, seed=7, **SPLIT_DEFAULTS)
    again = split_noniid_indices(dataset, num_clients=5, seed=7, **SPLIT_DEFAULTS)
    other = split_noniid_indices(dataset, num_clients=5, seed=8, **SPLIT_DEFAULTS)

    assert first == again
    assert first != other
