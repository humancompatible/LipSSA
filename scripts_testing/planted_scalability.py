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

Running. `python scripts_testing/planted_scalability.py --device cuda --grid large`
runs both sweeps and writes the JSON the notebook plots. Every (shape, method)
cell is cached through `experiments.cache`, so an interrupted run resumes where
it stopped and a grid can be spread over several short jobs; `--estimate`
prints the projected wall time of a grid without running it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.planted import make_planted_net
from experiments import cache
from experiments.nets import RowOutput
from hyperbox import Hyperbox
from other_methods.stochastic_approximation import StochasticApproximation
from other_methods.stochastic_approximation_ucb_dynamic import (
    StochasticApproximationUCBDynamic)

PRIMAL_NORM = 'l1'
DEFAULT_C = 0.8
DEFAULT_DEPTH1 = 1
HIT_TOL = 1e-6
CACHE_EXPERIMENT = 'planted_scalability'


def build(d, k, depth2, c=DEFAULT_C, depth1=DEFAULT_DEPTH1, gen_seed=0, device='cpu'):
    """Planted network, its domain, and the c_vector, ready for a solver.

    Returns (network, domain, c_vector, meta, n_neurons). The domain is the
    symmetric box [-1, 1]^d: the reaching region is the positive orthant, so a
    domain that is not symmetric about the origin changes rho and invalidates the
    analytic curve. The network is moved to `device`.
    """
    net, meta = make_planted_net(d=d, k=k, depth1=depth1, depth2=depth2,
                                 c=c, seed=gen_seed, dtype=torch.float32)
    domain = Hyperbox.build_linf_ball(np.zeros(d), 1.0)
    c_vector = np.ones(1 + (d - k))
    n_neurons = count_neurons(net, k, d, depth1, depth2)
    return RowOutput(net).to(device), domain, c_vector, meta, n_neurons


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


def make_solver(method, network, domain, c_vector, device='cpu'):
    """A solver of either kind, on the same network/domain/objective.

    Both expose the same interface -- `compute(max_iter, exact, tol, mode)`,
    `iteration_count`, `compute_time` -- so every measurement below is written
    once and run for each method.
    """
    if method == 'uniform':
        return StochasticApproximation(network, c_vector, domain,
                                       primal_norm=PRIMAL_NORM, device=device)
    if method == 'ucb':
        return StochasticApproximationUCBDynamic(
            network, c_vector, domain, c=UCB_C,
            partition_step=UCB_PARTITION_STEP, primal_norm=PRIMAL_NORM, device=device)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def _sync(device):
    """Wait for queued GPU work so wall-clock timings measure the computation."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def run_until_hit(method, network, domain, c_vector, c, seed, max_iter=20000, device='cpu'):
    """Running-max search, stopped as soon as the estimate reaches c.

    Returns dict with the recovered value, the number of iterations used, and the
    wall time of the whole search. `hit` is False if the budget ran out first.
    """
    torch.manual_seed(seed)
    sa = make_solver(method, network, domain, c_vector, device=device)
    value = float(sa.compute(max_iter=max_iter, exact=torch.tensor(float(c)),
                             tol=HIT_TOL, mode='Absolute'))
    _sync(device)
    hit = abs(value - c) <= HIT_TOL
    return {'value': value, 'evals': int(sa.iteration_count),
            'seconds': float(sa.compute_time), 'hit': bool(hit)}


def per_iter_cost(method, network, domain, c_vector, n_iters=100, seed=0, warmup=20, device='cpu'):
    """Mean wall time of one iteration, with the early stop disabled.

    For `uniform` an iteration is just the Jacobian-vector product: one forward
    and one backward pass. For `ucb` it also includes choosing a region by UCB
    score, drawing a point from that region, and folding the result into the
    tree statistics -- the bookkeeping that buys the adaptive partition. Comparing
    the two is the point: it prices what the smarter search costs per sample.

    A short warm-up runs first so the measurement excludes one-off allocation.
    """
    torch.manual_seed(seed)
    make_solver(method, network, domain, c_vector, device=device).compute(max_iter=warmup)
    _sync(device)

    sa = make_solver(method, network, domain, c_vector, device=device)
    t0 = time.time()
    sa.compute(max_iter=n_iters)
    _sync(device)
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

def _device_label(device):
    """Device string with the GPU model, so timings are attributable."""
    dev = torch.device(device)
    if dev.type == 'cuda':
        return f"cuda:{torch.cuda.get_device_name(dev)}"
    return dev.type


def sweep_shapes(shapes, k=5, c=DEFAULT_C, n_seeds=25, cost_iters=30,
                 methods=METHODS, device='cpu', verbose=True, force=False):
    """Grow the network along one axis and time both methods on it.

    `shapes` is a list of (d, depth2); pass a grid that varies one of the two and
    holds the other fixed, so an effect can be attributed to width or to depth
    rather than to "size". Each configuration is measured twice, once per method,
    on the same network: per-iteration cost with the early stop disabled, and the
    wall time of a full search that stops on reaching c.

    Each (shape, method) cell is cached by `experiments.cache` under
    `CACHE_EXPERIMENT`, keyed by everything that determines it including the
    device, so a re-run only computes what is missing.
    """
    records = []
    dev_label = _device_label(device)
    for d, depth2 in shapes:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c, device=device)
        n_params = sum(p.numel() for p in network.parameters())
        budget = budget_for(meta.rho)
        rec = {'d': d, 'depth2': depth2, 'k': k, 'c': c, 'rho': meta.rho,
               'n_neurons': n_neurons, 'n_params': int(n_params),
               'device': dev_label, 'methods': {}}
        for method in methods:
            key = dict(d=d, depth2=depth2, k=k, c=c, depth1=DEFAULT_DEPTH1, method=method,
                       n_seeds=n_seeds, cost_iters=cost_iters, device=dev_label,
                       primal_norm=PRIMAL_NORM)

            def cell():
                cost = per_iter_cost(method, network, domain, c_vector,
                                     n_iters=cost_iters, device=device)
                runs = [run_until_hit(method, network, domain, c_vector, c, seed=s,
                                      max_iter=budget, device=device) for s in range(n_seeds)]
                return {'per_iter_s': cost,
                        'values': [r['value'] for r in runs],
                        'evals': [r['evals'] for r in runs],
                        'seconds': [r['seconds'] for r in runs],
                        'hits': [r['hit'] for r in runs]}

            stored = cache.run_cached(CACHE_EXPERIMENT, key, cell, force=force)
            rec['methods'][method] = {f: stored[f] for f in
                                      ('per_iter_s', 'values', 'evals', 'seconds', 'hits')}
            if verbose:
                v = np.array(rec['methods'][method]['values'])
                print(f"d={d:6d} depth2={depth2:5d} params={n_params / 1e6:8.1f}M "
                      f"{method:>8s}  value/c in "
                      f"[{v.min()/c:.6f}, {v.max()/c:.6f}]  "
                      f"median iters="
                      f"{int(np.median(rec['methods'][method]['evals'])):5d}  "
                      f"per-iter={1e3 * stored['per_iter_s']:7.3f}ms  "
                      f"median run="
                      f"{np.median(rec['methods'][method]['seconds']):7.3f}s",
                      flush=True)
        records.append(rec)
        del network                    # free the weights before the next, larger shape
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    return records


def sweep_k(ks, d=256, depth2=4, c=DEFAULT_C, n_seeds=60, device='cpu', verbose=True):
    """Grow the needle's rarity at fixed network size.

    Evaluations to reach c should track the analytic mean 1/rho = 2^k.
    """
    records = []
    for k in ks:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c, device=device)
        budget = budget_for(meta.rho)
        runs = [run_until_hit('uniform', network, domain, c_vector, c, seed=s,
                              max_iter=budget, device=device) for s in range(n_seeds)]
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
WIDTH_GRID_LARGE = WIDTH_GRID + [(d, FIXED_DEPTH) for d in (8192, 16384)]
DEPTH_GRID_LARGE = DEPTH_GRID + [(FIXED_WIDTH, p) for p in (1024, 2048)]
GRIDS = {'default': (WIDTH_GRID, DEPTH_GRID),
         'large': (WIDTH_GRID_LARGE, DEPTH_GRID_LARGE)}
K_GRID = [3, 5, 8, 10, 12]
RESULTS = ROOT / 'scripts_testing' / 'planted_out' / 'scalability_methods.json'


def run_all(out=RESULTS, n_seeds=25, k=5, methods=METHODS, device='cpu', grid='default',
            force=False):
    """Run the two shape sweeps for every method and write the notebook's JSON.

    The notebook plots this file and does not compute; `n_seeds`, `grid` and
    `device` are the knobs worth turning. Cells already in the cache are reused
    unless `force`. The k sweep is not part of this -- it measures the instance,
    not scalability -- but `sweep_k` is still here.
    """
    import json
    width_grid, depth_grid = GRIDS[grid]
    print(f"== width sweep: d varies, depth2={FIXED_DEPTH}, k={k}, device={_device_label(device)} ==",
          flush=True)
    width = sweep_shapes(width_grid, k=k, n_seeds=n_seeds, methods=methods, device=device, force=force)
    print(f"\n== depth sweep: depth2 varies, d={FIXED_WIDTH}, k={k} ==", flush=True)
    depth = sweep_shapes(depth_grid, k=k, n_seeds=n_seeds, methods=methods, device=device, force=force)
    data = {'width': width, 'depth': depth, 'methods': list(methods),
            'device': _device_label(device), 'grid': grid}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"\nwrote {out}")
    return data


def estimate_runtime(shapes, k=5, n_seeds=25, c=DEFAULT_C, cost_iters=10,
                     methods=METHODS, device='cpu', verbose=True):
    """Project the wall time of `sweep_shapes(shapes, ...)` without running it.

    Builds each network, times a few iterations of each method, and scales by the
    expected iterations per run -- 1/rho, the mean of the geometric hitting time --
    times the number of seeds. Meant to be called before committing to a grid, so
    one that would run for an hour can be seen for what it is first.
    """
    total = 0.0
    for d, depth2 in shapes:
        network, domain, c_vector, meta, n_neurons = build(d, k, depth2, c=c, device=device)
        for method in methods:
            cost = per_iter_cost(method, network, domain, c_vector,
                                 n_iters=cost_iters, warmup=3, device=device)
            secs = n_seeds * cost / meta.rho
            total += secs
            if verbose:
                print(f"  d={d:6d} depth2={depth2:5d} {method:>8s}  "
                      f"per-iter={1e3 * cost:7.3f}ms  ->  {secs:8.1f}s", flush=True)
        del network
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--device', default='cpu', help="'cpu', 'cuda' or 'cuda:N'")
    p.add_argument('--grid', default='default', choices=sorted(GRIDS))
    p.add_argument('--seeds', type=int, default=25, help='searches per (shape, method)')
    p.add_argument('--k', type=int, default=5, help='planted width; sets rho = 2^-k')
    p.add_argument('--out', type=Path, default=RESULTS, help='JSON the notebook plots')
    p.add_argument('--estimate', action='store_true',
                   help='print the projected wall time of the grid and exit')
    p.add_argument('--force', action='store_true', help='recompute cells already in the cache')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()
    if args.estimate:
        for name, shapes in zip(('width', 'depth'), GRIDS[args.grid]):
            print(f"== {name} sweep ==")
            estimate_runtime(shapes, k=args.k, n_seeds=args.seeds, device=args.device)
    else:
        run_all(out=args.out, n_seeds=args.seeds, k=args.k, device=args.device,
                grid=args.grid, force=args.force)
