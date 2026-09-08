"""Contract of the FedAvg aggregation step."""

import pytest
import torch

from xray_fl.federated import fedavg


def test_fedavg_weights_updates_by_client_sample_count():
    updates = [
        {"num_samples": 1, "state_dict": {"weight": torch.tensor([0.0, 4.0])}},
        {"num_samples": 3, "state_dict": {"weight": torch.tensor([4.0, 0.0])}},
    ]

    averaged = fedavg(updates)

    assert torch.allclose(averaged["weight"], torch.tensor([3.0, 1.0]))


def test_fedavg_rejects_empty_and_zero_sample_updates():
    with pytest.raises(ValueError):
        fedavg([])

    with pytest.raises(ValueError):
        fedavg([{"num_samples": 0, "state_dict": {"weight": torch.zeros(1)}}])
