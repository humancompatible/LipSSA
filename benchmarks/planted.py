"""Generator for planted networks with a known local Lipschitz constant.

Builds F(x) = (F_1(u), F_2(v)) with x = (u, v), u = x[:k], v = x[k:], whose local
Lipschitz constant L_p(F) = c is known by construction for every p in {1, 2, inf}.

Two parallel blocks read disjoint slices of the input (nothing mixes u and v), so
the Jacobian is block-diagonal and its induced p-norm is the max of the two block
norms:
  - planted block F_1: narrow, exact Lipschitz constant c;
  - distractor block F_2: wide/deep, Lipschitz constant < c by construction.
Hence L_p(F) = max(c, L(F_2)) = c.

The planted block is a min-gadget: the estimate reaches c exactly on the positive
orthant of u, a region of known measure rho = 2^{-k}. Under coordinate-symmetric
sampling this gives the analytic miss probability Pr[Lhat_n < c] = (1 - rho)^n for
a running-max estimator over n samples.

Min-tree encoding
-----------------
ReLU is monotone non-decreasing, hence it commutes with the minimum:

    phi(min_j u_j) = min_j phi(u_j),      phi = ReLU.

So a single ReLU applied to the raw input is enough to make every value inside the
tree non-negative, and the tree can then carry one channel per value. There is no
need for the two-channel signed encoding value = phi(value) - phi(-value): no
negative intermediate value exists to encode. The only quantity inside the tree
that genuinely changes sign is the pairwise difference a - b, and it passes
through its own ReLU(a - b) -- an honestly ambiguous neuron that must stay.

Permutations do not change a multiset, so the sigma stack in front of the tree
leaves min_j unchanged; combined with the identity above, the block computes
c * ReLU(min_j u_j) regardless of depth1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from . import utils

# --- Named constants -------------------------------------------------------
DEFAULT_DTYPE = utils.DEFAULT_DTYPE
DEFAULT_SLOPE_DIST = (0.2, 0.95)      # U(a, b) for distractor slopes, b < 1 strictly
DEFAULT_BIRKHOFF_TERMS = 10           # K permutations per doubly-stochastic matrix

# Independent random-stream offsets (see utils.make_generator).
_OFF_SIGMA = 1        # planted permutation stack Sigma
_OFF_DIST_MATS = 2    # distractor doubly-stochastic matrices
_OFF_SLOPES = 3       # distractor slopes


@dataclass
class PlantedMeta:
    """Ground-truth metadata carried alongside the generated network."""
    true_lipschitz: float                 # = c
    rho: float                            # measure of the reaching region: 2^-k
    k: int
    d: int
    c: float
    seed: int
    reaching_region: str                  # human-readable description
    # Raw building blocks, exposed for inspection/testing (they are checked at a
    # different level than the final layer weights):
    A_raw: list = field(default_factory=list)       # doubly-stochastic mats BEFORE slope/c
    slopes: list = field(default_factory=list)       # per-layer slope vectors a_l
    c_absorbed_layer: str = ""            # where the scalar c is folded in


# =========================================================================
# Planted block: min-gadget   R^k -> R
# =========================================================================
def _build_min_tree_trunk(k: int, dtype: torch.dtype):
    """Layers reducing k *non-negative* channels to the single value min_j u_j.

    Computes the minimum with a genuine ReLU network (no torch.min), so the block
    stays a plain feed-forward net with explicit weight matrices. It uses
    min(a, b) = a - ReLU(a - b) over a binary tree of ceil(log2 k) levels. Because
    the caller guarantees non-negative inputs (see the module docstring), one
    channel per value suffices -- no signed two-channel encoding.

    Each level needs one ReLU layer:
      - layer A (before the ReLU) emits, per pair (a, b), the rows [a - b] and
        [a]; ReLU turns them into d = ReLU(a - b) and a itself (a >= 0). An odd
        tail value is carried by a single row [a].
      - layer B (after the ReLU) forms min(a, b) = a - d with one row per value.
        min(a, b) >= 0, so no ReLU is needed after it; instead of emitting layer B
        as its own linear map it is left *pending* and composed into the next
        level's layer A (or into the head), which keeps the block strictly
        alternating Linear -> ReLU and removes a layer of pass-through neurons.

    All weights are in {0, +-1} and non-trainable.

    Returns (nn.ModuleList of alternating (Linear, ReLU) layers, pending), where
    `pending` is the (1 x width) matrix the caller must fold into its head.
    """
    layers = nn.ModuleList()
    n = k
    pending = torch.eye(k, dtype=dtype)     # maps last ReLU output -> live values

    while n > 1:
        # --- Layer A: per pair emit [a - b] and [a]; odd tail emits [a]. ---
        rowsA = []
        layout = []          # semantics of layer-A outputs, in order
        for i in range(0, n, 2):
            if i + 1 < n:
                r = torch.zeros(n, dtype=dtype)
                r[i], r[i + 1] = 1.0, -1.0
                rowsA.append(r)                        # -> d = ReLU(a - b)
                ra = torch.zeros(n, dtype=dtype); ra[i] = 1.0
                rowsA.append(ra)                       # -> a  (a >= 0, ReLU is id)
                layout.append(("combine", len(rowsA) - 2))   # index of d; a at +1
            else:
                ra = torch.zeros(n, dtype=dtype); ra[i] = 1.0
                rowsA.append(ra)                       # -> carried tail value
                layout.append(("carry", len(rowsA) - 1))
        WA = torch.stack(rowsA, dim=0)
        layers.append(utils.fixed_linear(WA @ pending, dtype))
        layers.append(nn.ReLU())
        width_a = WA.shape[0]

        # --- Layer B: min(a, b) = a - d, one row per surviving value. ---
        rowsB = []
        for kind, base in layout:
            r = torch.zeros(width_a, dtype=dtype)
            if kind == "combine":
                d_idx, a_idx = base, base + 1
                r[a_idx], r[d_idx] = 1.0, -1.0         # min(a, b) = a - ReLU(a-b)
            else:
                r[base] = 1.0
            rowsB.append(r)
        pending = torch.stack(rowsB, dim=0)
        n = pending.shape[0]

    return layers, pending      # pending: (1 x width) once n == 1


class _MinTreeCore(nn.Module):
    """R^k -> R returning the tree's minimum, without the c scaling.

    Shares the trunk layers with the planted branch and reproduces everything the
    branch does except multiplying by c, so it evaluates ReLU(min_j u_j): the
    sigma stack in front of the tree already clamps negative coordinates, and
    ReLU commutes with the minimum. Exposed for inspection and testing.
    """

    def __init__(self, sigma: nn.ModuleList, trunk: nn.ModuleList,
                 pending: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.sigma = sigma
        self.trunk = trunk
        self.head = utils.fixed_linear(pending, dtype)   # (1 x width), c omitted

    def forward(self, u):
        h = u
        for layer in self.sigma:
            h = layer(h)
        for layer in self.trunk:
            h = layer(h)
        return self.head(h).squeeze(-1)


class _PlantedMinGadget(nn.Module):
    """R^k -> R,  F_1(u) = c * ReLU(min_j u_j).

    Sigma is a stack of (permutation + ReLU) layers that add depth without
    changing the norm; a permutation leaves the minimum alone, so the block's
    value does not depend on depth1. Sigma also establishes the invariant the min
    tree relies on -- its output is non-negative -- which is why depth1 == 0 still
    gets one (identity + ReLU) layer rather than no layer at all.

    The tree is followed by a Linear with c folded in and a final ReLU. ReLU(c t)
    = c ReLU(t) for c > 0, so this equals c * ReLU(min); folding c into the last
    linear keeps the whole block a plain ReLU net rather than an extra scalar
    multiply on the output.
    """

    def __init__(self, k, depth1, c, seed, dtype):
        super().__init__()
        self.k = k
        gen = utils.make_generator(seed, _OFF_SIGMA)
        sigma = nn.ModuleList()
        if depth1 > 0:
            for _ in range(depth1):
                P = utils.random_permutation_matrix(k, gen, dtype)
                sigma.append(utils.fixed_linear(P, dtype))
                sigma.append(nn.ReLU())
        else:
            # The tree requires non-negative inputs; keep one ReLU regardless.
            sigma.append(utils.fixed_linear(torch.eye(k, dtype=dtype), dtype))
            sigma.append(nn.ReLU())
        self.sigma = sigma

        trunk, pending = _build_min_tree_trunk(k, dtype)
        self.trunk = trunk
        # head with c absorbed, composed with the tree's pending last level.
        self.head = utils.fixed_linear(c * pending, dtype)
        self.head_relu = nn.ReLU()

        # standalone min view sharing the same layers (c omitted)
        self.min_tree_core = _MinTreeCore(sigma, trunk, pending, dtype)

    def forward(self, u):
        h = u
        for layer in self.sigma:
            h = layer(h)
        for layer in self.trunk:
            h = layer(h)
        h = self.head_relu(self.head(h))
        return h.squeeze(-1)          # scalar output R^k -> R


# =========================================================================
# Distractor block   R^{d-k} -> R^{d-k}
# =========================================================================
class _Distractor(nn.Module):
    """Plain-ReLU stack of doubly-stochastic layers with slopes folded in.

    Intended map: F_2(v) = A_L sigma_{L-1}(... sigma_1(A_1 v)) with a leaky-style
    activation sigma_l(z) = diag(a_l) ReLU(z), a_l ~ U(slope_dist). Because
    a > 0 => a ReLU(t) = ReLU(a t), each slope can be absorbed into the *next*
    layer's weight, keeping the block a plain ReLU net:
      W_1 = A_1, W_l = A_l @ diag(a_{l-1}) for l >= 2, plain ReLU between layers.
    c is folded into the last weight. Since a doubly-stochastic A has ||A||_2 <= 1
    and max(a) < 1, every slope-bearing layer is contractive (||W||_2 < 1), so
    L(F_2) < 1 with a margin, and < c after the last layer's c factor.
    """

    def __init__(self, m, depth2, c, seed, slope_dist, birkhoff_terms, dtype):
        super().__init__()
        assert depth2 >= 1
        gen_mat = utils.make_generator(seed, _OFF_DIST_MATS)
        gen_slope = utils.make_generator(seed, _OFF_SLOPES)
        low, high = slope_dist

        self.A_raw = [utils.birkhoff_doubly_stochastic(m, birkhoff_terms, gen_mat, dtype)
                      for _ in range(depth2)]
        # one slope vector per hidden ReLU (between layer l and l+1): depth2 - 1 of them
        self.slopes = [utils.uniform_slopes(m, low, high, gen_slope, dtype)
                       for _ in range(depth2 - 1)]

        layers = nn.ModuleList()
        for l in range(depth2):
            A = self.A_raw[l]
            W = A if l == 0 else A @ torch.diag(self.slopes[l - 1])
            if l == depth2 - 1:
                W = c * W                       # fold c into the last linear
            layers.append(utils.fixed_linear(W, dtype))
        self.layers = layers
        self.depth2 = depth2

    def forward(self, v):
        h = v
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l < self.depth2 - 1:            # ReLU between layers, none after last
                h = torch.relu(h)
        return h


# =========================================================================
# Full parallel network
# =========================================================================
class PlantedNet(nn.Module):
    """F(x) = concat(F_1(x[:k]), F_2(x[k:])) with disjoint inputs."""

    def __init__(self, planted: nn.Module, distractor: nn.Module, k: int, d: int):
        super().__init__()
        self.planted = planted
        self.distractor = distractor
        self.k = k
        self.d = d

    def forward(self, x):
        u = x[..., :self.k]
        v = x[..., self.k:]
        p_out = self.planted(u)
        if p_out.dim() == u.dim() - 1:         # scalar planted output -> add a dim
            p_out = p_out.unsqueeze(-1)
        d_out = self.distractor(v)
        return torch.cat([p_out, d_out], dim=-1)


def make_planted_net(d, k, depth1, depth2, c, seed,
                     slope_dist=DEFAULT_SLOPE_DIST,
                     birkhoff_terms=DEFAULT_BIRKHOFF_TERMS,
                     dtype=DEFAULT_DTYPE):
    """Build a planted network and its ground-truth metadata.

    Returns (net, meta) with meta.true_lipschitz == c. See the module docstring
    for the construction and the guarantees on L_p(F).
    """
    if not (0 < c <= 1):
        raise ValueError(f"c must satisfy 0 < c <= 1, got {c}")
    if not (1 <= k < d):
        raise ValueError(f"need 1 <= k < d, got k={k}, d={d}")
    if slope_dist[1] >= 1:
        raise ValueError(f"slope_dist upper bound must be < 1, got {slope_dist}")
    m = d - k

    planted = _PlantedMinGadget(k, depth1, c, seed, dtype)
    rho = 2.0 ** (-k)
    reaching = ("positive orthant u in (0,1]^k (measure 2^-k); "
                "argmin cell has ||J_F1||_p = 1")

    distractor = _Distractor(m, depth2, c, seed, slope_dist, birkhoff_terms, dtype)
    net = PlantedNet(planted, distractor, k, d)

    meta = PlantedMeta(
        true_lipschitz=float(c), rho=float(rho), k=k, d=d, c=float(c), seed=seed,
        reaching_region=reaching,
        A_raw=[A.clone() for A in distractor.A_raw],
        slopes=[s.clone() for s in distractor.slopes],
        c_absorbed_layer="last linear of each block (planted head + distractor layer L)",
    )
    return net, meta
