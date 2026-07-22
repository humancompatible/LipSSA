"""Generator for planted networks with a known local Lipschitz constant.

Builds F(x) = (F_1(u), F_2(v)) with x = (u, v), u = x[:k], v = x[k:], whose local
Lipschitz constant L_p(F) = c is known by construction for every p in {1, 2, inf}.

Two parallel blocks read DISJOINT slices of the input (nothing mixes u and v), so
the Jacobian is block-diagonal and its induced p-norm is the max of the two block
norms:
  - planted block F_1: narrow, exact Lipschitz constant c;
  - distractor block F_2: wide/deep, Lipschitz constant < c by construction.
Hence L_p(F) = max(c, L(F_2)) = c.

For the "min_gadget" planted block the estimate reaches c exactly on the positive
orthant of u, a region of known measure rho = 2^{-k}. Under coordinate-symmetric
sampling this gives the analytic miss probability Pr[Lhat_n < c] = (1 - rho)^n for
a running-max estimator over n samples.
"""

from __future__ import annotations

import math
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

PLANTED_MIN_GADGET = "min_gadget"
PLANTED_FLAWED = "flawed_permutation"


@dataclass
class PlantedMeta:
    """Ground-truth metadata carried alongside the generated network."""
    true_lipschitz: float                 # = c
    rho: float                            # measure of the reaching region: 2^-k or 1-2^-k
    k: int
    d: int
    c: float
    seed: int
    planted_style: str
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
    """Layers mapping raw u in R^k to a final two-channel pair (P, M) encoding
    min_j u_j = P - M, with P, M >= 0.

    This computes the minimum with a genuine ReLU network (no torch.min), so it
    stays a plain feed-forward ReLU net with explicit weight matrices. It uses:
      - min(a, b) = a - ReLU(a - b), applied over a binary tree of depth
        ceil(log2 k);
      - a two-channel encoding value = ReLU(value) - ReLU(-value) to carry
        possibly-negative intermediate values through ReLU layers.
    All weights are in {0, +-1} and non-trainable.

    Returns (nn.ModuleList of alternating (Linear, ReLU) layers, final width).
    """
    layers = nn.ModuleList()

    # --- Split layer: u_j -> (ReLU(u_j), ReLU(-u_j)) = (p_j, m_j). ---
    W_split = torch.zeros(2 * k, k, dtype=dtype)
    for j in range(k):
        W_split[2 * j, j] = 1.0        # p_j
        W_split[2 * j + 1, j] = -1.0   # m_j
    layers.append(utils.fixed_linear(W_split, dtype))
    layers.append(nn.ReLU())

    # live[i] = (p_idx, m_idx): channel indices of value i (value = p - m).
    live = [(2 * i, 2 * i + 1) for i in range(k)]
    width = 2 * k

    # --- Binary min tree, ceil(log2 k) levels, two Linear+ReLU layers each. ---
    while len(live) > 1:
        # Layer A: for each combined pair (a, b) emit [ReLU(a-b), pa, ma];
        #          for a carried tail value emit its pair unchanged.
        rowsA = []
        a_layout = []   # semantics of layer-A outputs, in order
        for i in range(0, len(live), 2):
            if i + 1 < len(live):
                (pa, ma), (pb, mb) = live[i], live[i + 1]
                # d = a - b = (pa - ma) - (pb - mb)
                r = torch.zeros(width, dtype=dtype)
                r[pa], r[ma], r[pb], r[mb] = 1.0, -1.0, -1.0, 1.0
                rowsA.append(r)                       # ReLU(d)
                rp = torch.zeros(width, dtype=dtype); rp[pa] = 1.0
                rowsA.append(rp)                      # pass pa (>=0) through ReLU
                rm = torch.zeros(width, dtype=dtype); rm[ma] = 1.0
                rowsA.append(rm)                      # pass ma (>=0) through ReLU
                a_layout.append(("combine", len(rowsA) - 3))  # index of ReLU(d)
            else:
                (p, m) = live[i]
                rp = torch.zeros(width, dtype=dtype); rp[p] = 1.0
                rowsA.append(rp)
                rm = torch.zeros(width, dtype=dtype); rm[m] = 1.0
                rowsA.append(rm)
                a_layout.append(("carry", len(rowsA) - 2))
        WA = torch.stack(rowsA, dim=0)
        layers.append(utils.fixed_linear(WA, dtype))
        layers.append(nn.ReLU())
        width_a = WA.shape[0]

        # Layer B: for each combined triple [ReLU(d), pa, ma] form the new pair
        #          (ReLU(minval), ReLU(-minval)), minval = pa - ma - ReLU(d);
        #          carried pairs pass through.
        rowsB = []
        new_live = []
        for kind, base in a_layout:
            if kind == "combine":
                rd, pa, ma = base, base + 1, base + 2
                rP = torch.zeros(width_a, dtype=dtype)
                rP[pa], rP[ma], rP[rd] = 1.0, -1.0, -1.0     # minval
                rowsB.append(rP)
                rM = torch.zeros(width_a, dtype=dtype)
                rM[pa], rM[ma], rM[rd] = -1.0, 1.0, 1.0      # -minval
                rowsB.append(rM)
            else:  # carry
                p, m = base, base + 1
                rp = torch.zeros(width_a, dtype=dtype); rp[p] = 1.0
                rowsB.append(rp)
                rm = torch.zeros(width_a, dtype=dtype); rm[m] = 1.0
                rowsB.append(rm)
            new_live.append((len(rowsB) - 2, len(rowsB) - 1))
        WB = torch.stack(rowsB, dim=0)
        layers.append(utils.fixed_linear(WB, dtype))
        layers.append(nn.ReLU())
        width = WB.shape[0]
        live = new_live

    return layers, width  # width == 2 (final pair), or 2 for k==1 too


class _MinTreeCore(nn.Module):
    """R^k -> R returning the *signed* min_j u_j (no final ReLU, no c scaling).

    Exposed for inspection/testing on inputs that include negative coordinates.
    Shares the trunk layers with the planted branch.
    """

    def __init__(self, trunk: nn.ModuleList, final_width: int, dtype: torch.dtype):
        super().__init__()
        self.trunk = trunk
        # head: (P, M) -> P - M = min
        W = torch.tensor([[1.0, -1.0]], dtype=dtype)
        assert final_width == 2
        self.head = utils.fixed_linear(W, dtype)

    def forward(self, u):
        h = u
        for layer in self.trunk:
            h = layer(h)
        return self.head(h).squeeze(-1)


class _PlantedMinGadget(nn.Module):
    """R^k -> R,  F_1(u) = c * ReLU(min_j (Sigma(u))_j).

    Sigma is a stack of depth1 (permutation + ReLU) layers that add depth without
    changing the norm; on the positive orthant it acts as the identity up to a
    coordinate permutation. It feeds the min tree, then a final Linear([c, -c])
    followed by ReLU, which equals c * ReLU(min) since c > 0 => ReLU(c t) =
    c ReLU(t). Folding c into the last linear keeps the whole block a plain ReLU
    net (rather than an extra scalar multiply on the output).
    """

    def __init__(self, k, depth1, c, seed, dtype):
        super().__init__()
        self.k = k
        gen = utils.make_generator(seed, _OFF_SIGMA)
        sigma = nn.ModuleList()
        for _ in range(depth1):
            P = utils.random_permutation_matrix(k, gen, dtype)
            sigma.append(utils.fixed_linear(P, dtype))
            sigma.append(nn.ReLU())
        self.sigma = sigma

        trunk, final_width = _build_min_tree_trunk(k, dtype)
        self.trunk = trunk
        # final head with c absorbed: (P, M) -> c*(P - M), then ReLU.
        Wc = torch.tensor([[c, -c]], dtype=dtype)
        self.head = utils.fixed_linear(Wc, dtype)
        self.head_relu = nn.ReLU()

        # standalone signed-min view sharing the same trunk (for T5)
        self.min_tree_core = _MinTreeCore(trunk, final_width, dtype)

    def forward(self, u):
        h = u
        for layer in self.sigma:
            h = layer(h)
        for layer in self.trunk:
            h = layer(h)
        h = self.head_relu(self.head(h))
        return h.squeeze(-1)          # scalar output R^k -> R


class _PlantedFlawedPermutation(nn.Module):
    """R^k -> R^k,  a pure-permutation alternative kept as a control construction.

    depth1 layers of (permutation + ReLU) then a final permutation scaled by c,
    i.e. F_1 = c * P_last @ phi(... phi(P_1 u)). Off the all-negative orthant the
    Jacobian is a partial permutation (a permutation with some rows deleted),
    which still has norm 1. So the reaching region here has measure 1 - 2^{-k}
    -- almost the whole domain -- inverting the difficulty knob relative to the
    min-gadget. This makes the estimator hit c almost immediately; it is useful
    precisely as a degenerate baseline to contrast with min_gadget.
    """

    def __init__(self, k, depth1, c, seed, dtype):
        super().__init__()
        self.k = k
        gen = utils.make_generator(seed, _OFF_SIGMA)
        layers = nn.ModuleList()
        for _ in range(depth1):
            P = utils.random_permutation_matrix(k, gen, dtype)
            layers.append(utils.fixed_linear(P, dtype))
            layers.append(nn.ReLU())
        # final permutation with c folded in, no trailing ReLU
        P_last = utils.random_permutation_matrix(k, gen, dtype)
        layers.append(utils.fixed_linear(c * P_last, dtype))
        self.layers = layers

    def forward(self, u):
        h = u
        for layer in self.layers:
            h = layer(h)
        return h


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
                     planted_style=PLANTED_MIN_GADGET,
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

    if planted_style == PLANTED_MIN_GADGET:
        planted = _PlantedMinGadget(k, depth1, c, seed, dtype)
        rho = 2.0 ** (-k)
        reaching = ("positive orthant u in (0,1]^k (measure 2^-k); "
                    "argmin cell has ||J_F1||_p = 1")
    elif planted_style == PLANTED_FLAWED:
        planted = _PlantedFlawedPermutation(k, depth1, c, seed, dtype)
        rho = 1.0 - 2.0 ** (-k)      # inverted measure: reaching region is almost everything
        reaching = ("complement of the all-negative orthant (measure 1-2^-k); "
                    "partial-permutation Jacobian of norm 1")
    else:
        raise ValueError(f"unknown planted_style {planted_style!r}")

    distractor = _Distractor(m, depth2, c, seed, slope_dist, birkhoff_terms, dtype)
    net = PlantedNet(planted, distractor, k, d)

    meta = PlantedMeta(
        true_lipschitz=float(c), rho=float(rho), k=k, d=d, c=float(c), seed=seed,
        planted_style=planted_style, reaching_region=reaching,
        A_raw=[A.clone() for A in distractor.A_raw],
        slopes=[s.clone() for s in distractor.slopes],
        c_absorbed_layer="last linear of each block (planted head + distractor layer L)",
    )
    return net, meta
