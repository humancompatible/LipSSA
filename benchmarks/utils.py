"""Shared helpers for the planted benchmark: deterministic random primitives,
doubly-stochastic matrix generation, and the Jacobian operator-norm estimator.
Pure PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# --- Named tolerances / constants -----------------------------------------
ROW_COL_SUM_ATOL = 1e-12          # tolerance on doubly-stochastic row/col sums (float64)
SPECTRAL_NORM_ATOL = 1e-10        # slack for the ||A||_2 <= 1 check
DEFAULT_DTYPE = torch.float64

# p-norms we support for the induced operator norm of the Jacobian.
SUPPORTED_P = (1, 2, math.inf)


# --- Deterministic random primitives --------------------------------------
def make_generator(seed: int, offset: int = 0) -> torch.Generator:
    """A CPU generator seeded deterministically (seed + offset).

    Separate offsets are used for the different random streams (permutations,
    Dirichlet weights, slopes) so the streams stay independent and reproducible.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(offset))
    return g


def random_permutation_matrix(k: int, generator: torch.Generator,
                              dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """A k x k permutation matrix P = eye(k)[perm].

    A permutation matrix only reorders coordinates, so ||P||_p = 1 for every
    induced p-norm.
    """
    perm = torch.randperm(k, generator=generator)
    return torch.eye(k, dtype=dtype)[perm]


def dirichlet_ones(n: int, generator: torch.Generator,
                   dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """Sample theta ~ Dirichlet(1, ..., 1) on the (n-1)-simplex.

    Implemented via Exponential(1) = -log(U(0,1)) normalised, so a torch.Generator
    can drive it reproducibly (torch.distributions has no generator argument).
    """
    u = torch.rand(n, generator=generator, dtype=dtype)
    exp = -torch.log(u)
    return exp / exp.sum()


def birkhoff_doubly_stochastic(m: int, n_terms: int, generator: torch.Generator,
                               dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """Doubly-stochastic m x m matrix as a convex combination of permutations.

    A = sum_i theta_i P_i with theta on the simplex and P_i permutation matrices.
    Such a matrix has non-negative entries with unit row and column sums, and
    since ||P_i||_p = 1, the triangle inequality gives ||A||_p <= sum_i theta_i
    = 1 for every p in {1, 2, inf}.
    """
    theta = dirichlet_ones(n_terms, generator, dtype=dtype)
    A = torch.zeros(m, m, dtype=dtype)
    for i in range(n_terms):
        A = A + theta[i] * random_permutation_matrix(m, generator, dtype=dtype)
    return A


def uniform_slopes(m: int, low: float, high: float, generator: torch.Generator,
                   dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """Per-neuron slopes a ~ U(low, high). Requiring high < 1 strictly keeps the
    slope-bearing layers contractive with a margin."""
    return low + (high - low) * torch.rand(m, generator=generator, dtype=dtype)


# --- Fixed (non-trainable) linear layer -----------------------------------
def fixed_linear(weight: torch.Tensor, dtype: torch.dtype = DEFAULT_DTYPE) -> nn.Linear:
    """A bias-free nn.Linear with fixed, non-trainable weight."""
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=False, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight.to(dtype))
    lin.weight.requires_grad_(False)
    return lin


# --- Jacobian operator norm -----------------------------------------------
def batched_jacobian(net: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Full Jacobian dF/dx for a batch of inputs, shape (B, out_dim, in_dim).

    Reverse-mode autodiff vectorised over the batch. A ReLU network is piecewise
    linear, so this is the exact Jacobian of the linear piece containing each x.
    """
    jac_fn = torch.func.vmap(torch.func.jacrev(net))
    return jac_fn(X)


def operator_norm(J: torch.Tensor, p) -> torch.Tensor:
    """Induced (alpha=beta=p) operator norm of each Jacobian in a batch.

    J: (B, out, in). Returns (B,). Closed forms for the induced matrix norms:
      p=1   -> max absolute column sum,
      p=inf -> max absolute row sum,
      p=2   -> largest singular value.
    """
    if p == 2:
        return torch.linalg.matrix_norm(J, ord=2)
    if p == math.inf:
        return J.abs().sum(dim=-1).amax(dim=-1)      # max row sum
    if p == 1:
        return J.abs().sum(dim=-2).amax(dim=-1)      # max col sum
    raise ValueError(f"unsupported p={p!r}; use one of {SUPPORTED_P}")


def jacobian_norms(net: nn.Module, X: torch.Tensor, p) -> torch.Tensor:
    """Convenience: ||J_net(x_i)||_p for every x_i in the batch X."""
    return operator_norm(batched_jacobian(net, X), p)
