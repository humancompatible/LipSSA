"""Run one sampling estimator on one network and return a JSON-ready record.

Every experiment calls `run_method` for each (method, seed) cell and stores the
record through `cache`. The record keeps the solver's improvement history, from
which `evals_to` and `best_so_far` derive iterations-to-tolerance and the
convergence curve without re-running anything.
"""

from __future__ import annotations

import numpy as np
import torch

from other_methods.stochastic_approximation import StochasticApproximation
from other_methods.stochastic_approximation_ucb_dynamic import StochasticApproximationUCBDynamic

# Default hyperparameters of the UCB search, the values used throughout the paper.
UCB_DEFAULTS = dict(c=15, partition_step=2, n0=10, split_rule='longest')

METHODS = ('uniform', 'ucb')


def make_solver(method, network, c_vector, domain, primal_norm='linf', device='cpu', **params):
    if method == 'uniform':
        assert not params, f"uniform search takes no parameters, got {params}"
        return StochasticApproximation(network, c_vector, domain, primal_norm=primal_norm, device=device)
    if method == 'ucb':
        kwargs = dict(UCB_DEFAULTS, **params)
        return StochasticApproximationUCBDynamic(network, c_vector, domain, primal_norm=primal_norm,
                                                 device=device, **kwargs)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def run_method(method, network, c_vector, domain, budget, seed, primal_norm='linf', device='cpu',
               exact=None, tol=None, mode='Relative', **params) -> dict:
    """One search of `budget` evaluations, seeded with `seed`.

    With `exact` and `tol` given the search stops early once the estimate is
    within `tol` of `exact` (`mode` 'Relative' means tol is a percentage, as in
    the solvers). Returns a dict with the final value, the number of evaluations
    used, wall time, and the improvement history [[iteration, seconds, value], ...].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = make_solver(method, network, c_vector, domain, primal_norm, device, **params)
    stop = {} if exact is None else dict(exact=torch.tensor(float(exact)), tol=tol, mode=mode)
    solver.compute(max_iter=budget, **stop)
    coords = getattr(solver, 'answer_coords', None)
    return {
        'method': method,
        'params': dict(UCB_DEFAULTS, **params) if method == 'ucb' else {},
        'seed': int(seed),
        'budget': int(budget),
        'primal_norm': primal_norm,
        'value': float(np.asarray(solver.value.detach().cpu() if torch.is_tensor(solver.value) else solver.value).squeeze()),
        'iterations': int(solver.iteration_count),
        'compute_time': float(solver.compute_time),
        'history': [[int(i), float(t), float(v)] for i, t, v in solver.history],
        'answer_coords': None if coords is None else np.asarray(coords).reshape(-1).tolist(),
    }


def evals_to(record: dict, target: float, frac: float = 0.99):
    """First (iteration, seconds) at which the running maximum reached frac * target.

    Returns (None, None) if the run never got there within its budget.
    """
    level = frac * target
    for it, t, v in record['history']:
        if v >= level:
            return it, t
    return None, None


def best_so_far(record: dict, grid) -> np.ndarray:
    """The running maximum evaluated at each iteration in `grid` (1-based).

    Iterations before the first improvement return 0; iterations past the run's
    last evaluation hold its final value.
    """
    grid = np.asarray(grid)
    out = np.zeros(grid.shape, dtype=float)
    for it, _, v in record['history']:
        out[grid >= it] = v
    return out
