"""Scalability of LipSSA on planted networks with a known Lipschitz constant.

The planted generator (`benchmarks/planted.py`) builds F = (F_1(u), F_2(v)) on
disjoint input slices, so the Jacobian is block diagonal and
L(F) = max(L(F_1), L(F_2)) = c is known exactly. The planted block reaches c on
the positive orthant of u, a region of measure rho = 2^-k. That gives two things
at once: a ground truth to check the estimate against, and an analytic prediction
Pr[Lhat_n < c] = (1 - rho)^n for a running-max estimator over n uniform samples.

What this measures. The number of samples a running-max estimator needs is set by
rho, hence by the planted width k, and does not depend on how large the rest of
the network is; the cost of one sample is one forward and one backward pass, which
does grow with the network. Three sweeps separate those, each moving one thing:

  WIDTH_GRID  widens the distractor at fixed depth and fixed k
  DEPTH_GRID  deepens the distractor at fixed width and fixed k
  K_GRID      rarefies the needle at fixed network

The first two are deliberately separate rather than one grid growing width and
depth together: a single mixed grid cannot say whether an effect came from width
or from depth, and both are needed to show the sample count responds to neither.

Objective. Every run computes sup_x ||J_F(x)^T c||_inf, i.e. `primal_norm='l1'`
with a fixed c_vector of ones.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.planted import make_planted_net
from hyperbox import Hyperbox
from other_methods.stochastic_approximation import StochasticApproximation
from other_methods.stochastic_approximation_ucb_dynamic import (
    StochasticApproximationUCBDynamic)

PRIMAL_NORM = 'l1'
DEFAULT_C = 0.8
DEFAULT_DEPTH1 = 1
HIT_TOL = 1e-6


class _RowOutput(nn.Module):
    """Present a network's output as a 1 x m matrix.

    `StochasticApproximation.f` projects with `.mv(c_vector)`, which needs a
    matrix; the planted network returns a plain vector for a single input.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(0) if out.dim() == 1 else out


def build(d, k, depth2, c=DEFAULT_C, depth1=DEFAULT_DEPTH1, gen_seed=0):
    """Planted network, its domain, and the c_vector, ready for a solver.

    Returns (network, domain, c_vector, meta, n_neurons). The domain is the
    symmetric box [-1, 1]^d: the reaching region is the positive orthant, so a
    domain that is not symmetric about the origin changes rho and invalidates the
    analytic curve.
    """
    net, meta = make_planted_net(d=d, k=k, depth1=depth1, depth2=depth2,
                                 c=c, seed=gen_seed, dtype=torch.float32)
    domain = Hyperbox.build_linf_ball(np.zeros(d), 1.0)
    c_vector = np.ones(1 + (d - k))
    n_neurons = count_neurons(net, k, d, depth1, depth2)
    return _RowOutput(net), domain, c_vector, meta, n_neurons


def count_neurons(net, k, d, depth1, depth2):
    """Hidden units in the whole network: distractor plus planted block."""
    n = (d - k) * (depth2 - 1)                       # distractor hidden layers
    n += k * max(depth1, 1)                          # sigma stack
    width = k
    while width > 1:                                 # min tree, one ReLU per level
        n += width // 2 * 2 + width % 2
        width = (width + 1) // 2
    return n + 1                                     # head ReLU


METHODS = ('uniform', 'ucb')
UCB_C = 15              # exploration coefficient
UCB_PARTITION_STEP = 2  # first split at iteration 2, then 4, 8, 16, ...


def make_solver(method, network, domain, c_vector):
    """A solver of either kind, on the same network/domain/objective.

    Both expose the same interface -- `compute(max_iter, exact, tol, mode)`,
    `iteration_count`, `compute_time` -- so every measurement below is written
    once and run for each method.
    """
    if method == 'uniform':
        return StochasticApproximation(network, c_vector, domain,
                                       primal_norm=PRIMAL_NORM)
    if method == 'ucb':
        return StochasticApproximationUCBDynamic(
            network, c_vector, domain, c=UCB_C,
            partition_step=UCB_PARTITION_STEP, primal_norm=PRIMAL_NORM)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def run_until_hit(method, network, domain, c_vector, c, seed, max_iter=20000):
    """Running-max search, stopped as soon as the estimate reaches c.

    Returns dict with the recovered value, the number of iterations used, and the
    wall time of the whole search. `hit` is False if the budget ran out first.
    """
    torch.manual_seed(seed)
    sa = make_solver(method, network, domain, c_vector)
    value = float(sa.compute(max_iter=max_iter, exact=torch.tensor(float(c)),
                             tol=HIT_TOL, mode='Absolute'))
    hit = abs(value - c) <= HIT_TOL
    return {'value': value, 'evals': int(sa.iteration_count),
            'seconds': float(sa.compute_time), 'hit': bool(hit)}


def per_iter_cost(method, network, domain, c_vector, n_iters=100, seed=0, warmup=20):
    """Mean wall time of one iteration, with the early stop disabled.

    For `uniform` an iteration is just the Jacobian-vector product: one forward
    and one backward pass. For `ucb` it also includes choosing a region by UCB
    score, drawing from that region's point pool, and folding the result into the
    tree statistics -- the bookkeeping that buys the adaptive partition. Comparing
    the two is the point: it prices what the smarter search costs per sample.

    A short warm-up runs first so the measurement excludes one-off allocation.
    """
    torch.manual_seed(seed)
    make_solver(method, network, domain, c_vector).compute(max_iter=warmup)

    sa = make_solver(method, network, domain, c_vector)
    t0 = time.time()
    sa.compute(max_iter=n_iters)
    return (time.time() - t0) / n_iters


def budget_for(rho, factor=50):
    """Sample budget generous enough that a miss is a measurement, not a cutoff.

    The running max misses after n draws with probability (1 - rho)^n, so
    n = factor / rho leaves e^-factor. Kept finite because the budget is also
    pre-allocated as an n x d tensor.
    """
    return int(np.ceil(factor / rho))


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def sweep_shapes(shapes, k=5, c=DEFAULT_C, n_seeds=25, cost_iters=30,
                 methods=METHODS, verbose=True):
    """Grow the network along one axis and time both methods on it.

    `shapes` is a list of (d, depth2); pass a grid that varies one of the two and
    holds the other fixed, so an effect can be attributed to width or to depth
    rather than to "size". Each configuration is measured twice, once per method,
    on the same network: per-iteration cost with the early stop disabled, and the
    wall time of a full search that stops on reaching c.
    """
    records = []
    for d, depth2 in shapes:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c)
        n_params = sum(p.numel() for p in network.parameters())
        budget = budget_for(meta.rho)
        rec = {'d': d, 'depth2': depth2, 'k': k, 'c': c, 'rho': meta.rho,
               'n_neurons': n_neurons, 'n_params': int(n_params), 'methods': {}}
        for method in methods:
            cost = per_iter_cost(method, network, domain, c_vector,
                                 n_iters=cost_iters)
            runs = [run_until_hit(method, network, domain, c_vector, c, seed=s,
                                  max_iter=budget) for s in range(n_seeds)]
            rec['methods'][method] = {
                'per_iter_s': cost,
                'values': [r['value'] for r in runs],
                'evals': [r['evals'] for r in runs],
                'seconds': [r['seconds'] for r in runs],
                'hits': [r['hit'] for r in runs]}
            if verbose:
                v = np.array(rec['methods'][method]['values'])
                print(f"d={d:6d} depth2={depth2:5d} params={n_params / 1e6:8.1f}M "
                      f"{method:>8s}  value/c in "
                      f"[{v.min()/c:.6f}, {v.max()/c:.6f}]  "
                      f"median iters="
                      f"{int(np.median(rec['methods'][method]['evals'])):5d}  "
                      f"per-iter={1e3 * cost:7.3f}ms  "
                      f"median run="
                      f"{np.median(rec['methods'][method]['seconds']):7.3f}s",
                      flush=True)
        records.append(rec)
    return records


def sweep_k(ks, d=256, depth2=4, c=DEFAULT_C, n_seeds=60, verbose=True):
    """Grow the needle's rarity at fixed network size.

    Evaluations to reach c should track the analytic mean 1/rho = 2^k.
    """
    records = []
    for k in ks:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c)
        budget = budget_for(meta.rho)
        runs = [run_until_hit('uniform', network, domain, c_vector, c, seed=s,
                              max_iter=budget) for s in range(n_seeds)]
        rec = {'k': k, 'd': d, 'depth2': depth2, 'c': c, 'rho': meta.rho,
               'n_neurons': n_neurons,
               'n_params': int(sum(p.numel() for p in network.parameters())),
               'evals': [r['evals'] for r in runs],
               'hits': [r['hit'] for r in runs],
               'values': [r['value'] for r in runs]}
        records.append(rec)
        if verbose:
            hit = np.mean(rec['hits'])
            print(f"k={k:3d} rho=2^-{k}={meta.rho:.2e}  median evals="
                  f"{int(np.median(rec['evals'])):7d}  (1/rho={1 / meta.rho:.0f})  "
                  f"hit rate={hit:.0%}", flush=True)
    return records


# One axis moves per grid, so an effect can be attributed to width or to depth.
FIXED_DEPTH, FIXED_WIDTH = 8, 512
WIDTH_GRID = [(d, FIXED_DEPTH) for d in (128, 256, 512, 1024, 2048, 4096)]
DEPTH_GRID = [(FIXED_WIDTH, p) for p in (4, 8, 16, 32, 64, 128, 256, 512)]
K_GRID = [3, 5, 8, 10, 12]
RESULTS = ROOT / 'scripts_testing' / 'planted_out' / 'scalability.json'


def run_all(out=RESULTS, n_seeds=25, k=5, methods=METHODS):
    """Run the two shape sweeps for every method and cache them.

    The notebook plots this file and does not compute; `n_seeds` and the grids
    above are the knobs worth turning. The k sweep is not part of this -- it
    measures the instance, not scalability -- but `sweep_k` is still here.
    """
    import json
    print(f"== width sweep: d varies, depth2={FIXED_DEPTH}, k={k} ==", flush=True)
    width = sweep_shapes(WIDTH_GRID, k=k, n_seeds=n_seeds, methods=methods)
    print(f"\n== depth sweep: depth2 varies, d={FIXED_WIDTH}, k={k} ==", flush=True)
    depth = sweep_shapes(DEPTH_GRID, k=k, n_seeds=n_seeds, methods=methods)
    data = {'width': width, 'depth': depth, 'methods': list(methods)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"\nwrote {out}")
    return data


def estimate_runtime(shapes, k=5, n_seeds=25, c=DEFAULT_C, cost_iters=10,
                     methods=METHODS, verbose=True):
    """Project the wall time of `sweep_shapes(shapes, ...)` without running it.

    Builds each network, times a few iterations of each method, and scales by the
    expected iterations per run -- 1/rho, the mean of the geometric hitting time --
    times the number of seeds. Meant to be called before committing to a grid, so
    one that would run for an hour can be seen for what it is first.
    """
    total = 0.0
    for d, depth2 in shapes:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c)
        for method in methods:
            cost = per_iter_cost(method, network, domain, c_vector,
                                 n_iters=cost_iters, warmup=3)
            secs = n_seeds * cost / meta.rho
            total += secs
            if verbose:
                print(f"  d={d:6d} depth2={depth2:5d} {method:>8s}  "
                      f"per-iter={1e3 * cost:7.3f}ms  ->  {secs:8.1f}s", flush=True)
    if verbose:
        print(f"  estimated total: {total:.0f}s ({total / 60:.1f} min) for "
              f"{n_seeds} seeds x {len(shapes)} shapes x {len(methods)} methods")
    return total


def miss_probability(evals, hits, grid):
    """Empirical Pr[Lhat_n < c] from the per-seed evaluation counts.

    A seed whose running max first reached c after e evaluations contributes a
    miss at every n < e; a seed that never hit within budget misses everywhere.
    """
    e = np.asarray(evals, dtype=float)
    e[~np.asarray(hits)] = np.inf
    return np.array([(e > n).mean() for n in grid])


if __name__ == '__main__':
    run_all()
