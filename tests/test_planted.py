"""Correctness tests for the planted-instance generator. All tests run in float64.

Each test checks one guarantee of the construction (block-diagonal Jacobian,
doubly-stochastic layers, the exact ground-truth constant, etc.).
"""

import math

import pytest
import torch

from benchmarks import utils
from benchmarks.planted import (
    PLANTED_FLAWED,
    PLANTED_MIN_GADGET,
    make_planted_net,
)

torch.manual_seed(0)

DTYPE = torch.float64
PS = (1, 2, math.inf)                 # induced norms to check

# tolerances (named, no magic numbers)
BLOCK_ATOL = 1e-12                     # off-diagonal Jacobian block == 0
DS_SUM_ATOL = utils.ROW_COL_SUM_ATOL   # doubly-stochastic row/col sums
SPEC_ATOL = utils.SPECTRAL_NORM_ATOL   # ||A||_2 <= 1 slack
STRICT_MARGIN = 1e-12                  # strict-inequality slack
GT_ATOL = 1e-9                         # ground-truth ||J||_p == c
AMIN_ATOL = 1e-10                      # min-tree vs torch.amin
JAC0_ATOL = 1e-10                      # off-orthant Jacobian == 0

# small-scale grids
KS = [2, 3, 5]
DS = [6, 20]
DEPTH1S = [0, 3, 17]
DEPTH2S = [2, 5]


def _orthant_u(n, k):
    """u samples strictly inside the positive orthant (0, 1]^k."""
    return torch.rand(n, k, dtype=DTYPE) * 0.9 + 0.05


def _full_input(n, d, k):
    """x = (u, v) with u on the positive orthant and v in [-1, 1]^{d-k}."""
    x = torch.empty(n, d, dtype=DTYPE)
    x[:, :k] = _orthant_u(n, k)
    x[:, k:] = torch.rand(n, d - k, dtype=DTYPE) * 2 - 1
    return x


# ----------------------------------------------------------------------- T1
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("d", DS)
def test_T1_jacobian_block_diagonal(k, d):
    """Disjoint inputs => Jacobian block-diagonal: d(planted)/dv = 0 and
    d(distractor)/du = 0."""
    if k >= d:
        pytest.skip("need k < d")
    net, _ = make_planted_net(d=d, k=k, depth1=3, depth2=3, c=1.0, seed=1, dtype=DTYPE)
    x = torch.rand(16, d, dtype=DTYPE) * 2 - 1
    J = utils.batched_jacobian(net, x)          # (B, out, d)
    p_out = 1                                    # min_gadget planted output is scalar
    assert J[:, :p_out, k:].abs().max() < BLOCK_ATOL   # d(planted)/dv
    assert J[:, p_out:, :k].abs().max() < BLOCK_ATOL   # d(distractor)/du


# ----------------------------------------------------------------------- T2
@pytest.mark.parametrize("c", [0.5, 1.0])
@pytest.mark.parametrize("depth2", DEPTH2S)
def test_T2_doubly_stochastic_and_layer_norms(c, depth2):
    """(a) raw A doubly-stochastic with ||A||_2 <= 1; (b) slope-bearing layer
    weights B = A@diag(a) have ||B||_2 < 1 strictly (last, c-scaled layer: <= c).
    These are distinct objects checked at distinct levels -- the raw matrices vs
    the final layer weights -- so iterating over parameters alone is not enough."""
    d, k = 20, 5
    net, meta = make_planted_net(d=d, k=k, depth1=3, depth2=depth2, c=c, seed=2, dtype=DTYPE)

    # (a) raw doubly-stochastic matrices
    for A in meta.A_raw:
        assert A.min() >= 0
        assert (A.sum(dim=1) - 1).abs().max() < DS_SUM_ATOL
        assert (A.sum(dim=0) - 1).abs().max() < DS_SUM_ATOL
        assert torch.linalg.matrix_norm(A, 2) <= 1 + SPEC_ATOL

    # (b) actual layer weights
    layers = net.distractor.layers
    for l, layer in enumerate(layers):
        n2 = torch.linalg.matrix_norm(layer.weight, 2).item()
        is_last = l == len(layers) - 1
        if l == 0 and not is_last:
            # layer 0 is a bare doubly-stochastic matrix: ||.||_2 == 1 exactly.
            assert abs(n2 - 1.0) < SPEC_ATOL
        elif is_last:
            # c folded in: B = c * A @ diag(a) => ||B||_2 <= c (strict when a<1).
            assert n2 < c - STRICT_MARGIN if depth2 > 1 else n2 <= c + SPEC_ATOL
        else:
            assert n2 < 1.0 - STRICT_MARGIN      # slope-bearing hidden layer contracts


# ----------------------------------------------------------------------- T3
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("depth1", DEPTH1S)
def test_T3_planted_orthant_jacobian_is_argmin(k, depth1):
    """min_gadget on the positive orthant: Jacobian of F_1 (c=1) equals
    e_{argmin(u)}^T exactly, and its norm is 1 in every p."""
    net, _ = make_planted_net(d=k + 6, k=k, depth1=depth1, depth2=2, c=1.0, seed=5, dtype=DTYPE)
    u = _orthant_u(64, k)
    J = utils.batched_jacobian(net.planted, u)   # (B, k)
    argmin = u.argmin(dim=-1)
    expected = torch.nn.functional.one_hot(argmin, k).to(DTYPE)
    assert (J - expected).abs().max() < GT_ATOL   # pointwise argmin routing
    for p in PS:
        norms = utils.operator_norm(J.unsqueeze(1), p)   # treat as 1 x k
        assert (norms - 1.0).abs().max() < GT_ATOL


# ----------------------------------------------------------------------- T4
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("depth1", DEPTH1S)
def test_T4_planted_off_orthant_jacobian_zero(k, depth1):
    """min_gadget off the positive orthant (>=1 negative coord): F_1 is locally
    constant, so its Jacobian is identically 0 (independent of the ReLU'(0)
    convention)."""
    net, _ = make_planted_net(d=k + 6, k=k, depth1=depth1, depth2=2, c=1.0, seed=6, dtype=DTYPE)
    u = torch.rand(64, k, dtype=DTYPE)
    u[:, 0] = -torch.rand(64, dtype=DTYPE) - 0.05   # force a strictly negative coord
    J = utils.batched_jacobian(net.planted, u)
    assert J.abs().max() < JAC0_ATOL


# ----------------------------------------------------------------------- T5
@pytest.mark.parametrize("k", KS)
def test_T5_min_tree_equals_amin(k):
    """The min-tree core (before final ReLU) equals torch.amin, including inputs
    with negative coordinates."""
    net, _ = make_planted_net(d=k + 6, k=k, depth1=0, depth2=2, c=1.0, seed=7, dtype=DTYPE)
    u = torch.rand(200, k, dtype=DTYPE) * 2 - 1
    got = net.planted.min_tree_core(u)
    assert (got - torch.amin(u, dim=-1)).abs().max() < AMIN_ATOL


# ----------------------------------------------------------------------- T6
@pytest.mark.parametrize("c", [0.5, 1.0])
@pytest.mark.parametrize("k", KS)
def test_T6_full_network_ground_truth(c, k):
    """Full network: on the orthant ||J_F(x)||_p == c for p in {1,2,inf} and any
    v; off the orthant ||J_F(x)||_p < c strictly. The max-of-blocks norm identity
    holds even though the planted block is rectangular (1 x k)."""
    d = 20
    net, _ = make_planted_net(d=d, k=k, depth1=5, depth2=4, c=c, seed=8, dtype=DTYPE)
    x = _full_input(48, d, k)
    for p in PS:
        norms = utils.jacobian_norms(net, x, p)
        assert (norms - c).abs().max() < GT_ATOL

    x_off = x.clone()
    x_off[:, 0] = -torch.rand(48, dtype=DTYPE) - 0.05
    for p in PS:
        norms = utils.jacobian_norms(net, x_off, p)
        assert norms.max() < c - GT_ATOL


# ----------------------------------------------------------------------- T7
@pytest.mark.parametrize("k", KS)
def test_T7_flawed_permutation_defect(k):
    """Control construction 'flawed_permutation': samples u OFF the orthant but
    with >=1 positive coordinate still have ||J_{F1}(u)||_2 == 1. This test is
    EXPECTED TO PASS -- it pins down the intended (degenerate) property of this
    construction, namely that its reaching region is almost the whole domain."""
    net, meta = make_planted_net(d=k + 6, k=k, depth1=3, depth2=2, c=1.0, seed=9,
                                 planted_style=PLANTED_FLAWED, dtype=DTYPE)
    assert meta.rho == pytest.approx(1 - 2.0 ** (-k))
    # u with exactly the first coord negative, rest positive -> >=1 positive coord
    u = torch.rand(64, k, dtype=DTYPE) * 0.9 + 0.05
    u[:, 0] = -torch.rand(64, dtype=DTYPE) - 0.05
    J = utils.batched_jacobian(net.planted, u)       # (B, k, k)
    norms = utils.operator_norm(J, 2)
    assert (norms - 1.0).abs().max() < GT_ATOL


# ----------------------------------------------------------------------- T8
@pytest.mark.parametrize("style,expected_rho", [
    (PLANTED_MIN_GADGET, lambda k: 2.0 ** (-k)),
    (PLANTED_FLAWED, lambda k: 1 - 2.0 ** (-k)),
])
def test_T8_empirical_rho(style, expected_rho):
    """Empirical fraction of samples with ||J_F|| >= c-eps matches the claimed
    measure rho of the reaching region within a binomial confidence interval."""
    d, k, N = 8, 3, 20000
    net, meta = make_planted_net(d=d, k=k, depth1=3, depth2=3, c=1.0, seed=11,
                                 planted_style=style, dtype=DTYPE)
    X = torch.rand(N, d, dtype=DTYPE) * 2 - 1
    norms = utils.jacobian_norms(net, X, 2)
    frac = (norms >= 1.0 - 1e-9).double().mean().item()

    p = expected_rho(k)
    assert meta.rho == pytest.approx(p)
    se = math.sqrt(p * (1 - p) / N)
    assert abs(frac - p) < 4 * se      # ~4-sigma binomial band


# ----------------------------------------------------------------------- T9
@pytest.mark.parametrize("style", [PLANTED_MIN_GADGET, PLANTED_FLAWED])
def test_T9_determinism(style):
    """Two calls with the same seed produce bitwise-identical state_dicts."""
    kwargs = dict(d=20, k=5, depth1=4, depth2=3, c=0.7, seed=123,
                  planted_style=style, dtype=DTYPE)
    net_a, _ = make_planted_net(**kwargs)
    net_b, _ = make_planted_net(**kwargs)
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key
