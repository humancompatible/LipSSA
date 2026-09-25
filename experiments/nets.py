"""Network factories shared by the experiments.

Each factory returns (network, domain, c_vector), exactly what the solvers and
LipMIP take.
Training is seeded so a (spec, seed) pair always yields the same network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from hyperbox import Hyperbox
from relu_nets import ReLUNet
import neural_nets.data_loaders as data_loaders
import neural_nets.train as train

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / 'datasets'


def synthetic_target(X):
    """The regression target of the synthetic benchmark: sin(||x||_2) + 0.1 x_1."""
    return torch.sin(X.norm(dim=1, keepdim=True)) + 0.1 * X[:, :1]


def train_synthetic(arch, seed, radius=1.0, n_samples=4096, epochs=1500, lr=1e-3):
    """ReLUNet `arch` fit to `synthetic_target` on uniform points of [-radius, radius]^d.

    Domain is that same box; c_vector is all ones over the outputs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ReLUNet(arch)
    d = arch[0]
    X = (torch.rand(n_samples, d) * 2 - 1) * radius
    y = synthetic_target(X)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss(net(X), y).backward()
        opt.step()
    net.eval()
    domain = Hyperbox.build_custom_hypercube(d, 0, radius)
    return net, domain, torch.ones(arch[-1])


def train_mnist(arch, seed, digits=(1, 7), epochs=5, lr=1e-3, batch_size=64, dataset_dir=DATASET_DIR):
    """ReLUNet `arch` (784 -> ... -> 2) trained as a binary classifier on two MNIST digits.

    Domain is the pixel box [0, 1]^784; c_vector = [1, -1] projects onto the
    logit difference, the quantity whose Lipschitz constant bounds the margin.
    """
    assert arch[0] == 784 and arch[-1] == 2, f"binary MNIST net must be [784, ..., 2], got {arch}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ReLUNet(arch)
    loaders = [data_loaders.load_mnist_data(split, digits=list(digits), batch_size=batch_size,
                                            shuffle=True, dataset_dir=str(dataset_dir))
               for split in ('train', 'val')]
    params = train.TrainParameters(*loaders, epochs, test_after_epoch=1,
                                   optimizer=torch.optim.Adam(net.parameters(), lr=lr))
    train.training_loop(net, params)
    net.eval()
    return net, Hyperbox.build_unit_hypercube(784), torch.tensor([1.0, -1.0])


class RowOutput(nn.Module):
    """Present a network's output as a 1 x m matrix.

    The solvers project with `.mv(c_vector)`, which needs a matrix; the planted
    network returns a plain vector for a single input.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(0) if out.dim() == 1 else out
