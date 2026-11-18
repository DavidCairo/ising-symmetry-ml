"""Tests for ising_ml.data, ising_ml.models and ising_ml.train (Days 3-4)."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ising_ml.data import IsingDataset, generate_dataset
from ising_ml.models import PlainCNN, Z2InvariantCNN
from ising_ml.train import evaluate, predict_proba, train_one_epoch

MODELS = [PlainCNN, Z2InvariantCNN]


def random_batch(B=8, L=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (B, 1, L, L), generator=g) * 2 - 1).float()


def test_generate_dataset():
    temps = np.array([1.5, 3.5])
    data = generate_dataset(8, temps, n_per_T=4, seed=0, n_therm=20, n_between=2)
    assert data["spins"].shape == (8, 8, 8)
    assert data["T"].shape == data["label"].shape == (8,)
    assert set(data["label"][data["T"] < 2.0]) == {1}
    assert set(data["label"][data["T"] > 3.0]) == {0}


def test_dataset_item_format():
    data = {
        "spins": np.ones((3, 8, 8), dtype=np.int8),
        "T": np.array([1.0, 2.0, 3.0]),
        "label": np.array([1, 1, 0]),
    }
    ds = IsingDataset(data)
    assert len(ds) == 3
    x, y = ds[2]
    assert x.shape == (1, 8, 8) and x.dtype == torch.float32
    assert y.shape == () and y.dtype == torch.float32 and y.item() == 0.0


@pytest.mark.parametrize("Model", MODELS)
@pytest.mark.parametrize("L", [8, 16])
def test_output_shape_any_L(Model, L):
    out = Model()(random_batch(L=L))
    assert out.shape == (8,)


@pytest.mark.parametrize("Model", MODELS)
def test_translation_invariance(Model):
    torch.manual_seed(0)
    model = Model().eval()
    x = random_batch()
    shifted = torch.roll(x, shifts=(3, 5), dims=(2, 3))
    torch.testing.assert_close(model(x), model(shifted), rtol=1e-4, atol=1e-5)


def test_z2_invariance_is_exact():
    torch.manual_seed(0)
    model = Z2InvariantCNN().eval()
    x = random_batch()
    torch.testing.assert_close(model(x), model(-x), rtol=1e-5, atol=1e-6)


def test_plain_cnn_is_not_z2_invariant():
    # sanity check that the Z2 test above is meaningful
    torch.manual_seed(0)
    model = PlainCNN().eval()
    x = random_batch()
    assert not torch.allclose(model(x), model(-x))


@pytest.mark.parametrize("Model", MODELS)
def test_training_reduces_loss(Model):
    # an easy problem: all-up/all-down (ordered) vs random (disordered)
    torch.manual_seed(0)
    L, n = 8, 64
    ordered = torch.ones(n, 1, L, L) * (torch.randint(0, 2, (n, 1, 1, 1)) * 2 - 1)
    disordered = random_batch(B=n, L=L, seed=1)
    x = torch.cat([ordered, disordered])
    y = torch.cat([torch.ones(n), torch.zeros(n)])
    loader = DataLoader(list(zip(x, y, strict=True)), batch_size=32, shuffle=True)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    before = evaluate(model, loader, loss_fn, device)["loss"]
    for _ in range(30):
        train_one_epoch(model, loader, opt, loss_fn, device)
    after = evaluate(model, loader, loss_fn, device)
    assert after["loss"] < before
    assert after["accuracy"] > 0.95

    p = predict_proba(model, x, device)
    assert p.shape == (2 * n,)
    assert torch.all((p >= 0) & (p <= 1))
