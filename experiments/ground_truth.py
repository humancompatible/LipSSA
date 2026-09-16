"""Exact reference values from LipMIP, with the timeout case reported honestly.

LipMIP returns `model.objBound` as `.value`. When the solve finishes that is the
optimum; when it hits the time limit it is only a proved upper bound, and the
solver's own best point (the incumbent, `model.ObjVal`) is a separate, lower
number. A comparison must not mix the two: the incumbent is a lower bound like
any sampling estimate, the bound is an upper bound. `lipmip_reference` returns
both and says which situation occurred.
"""

from __future__ import annotations

import math

import numpy as np

from lipMIP import LipMIP

GUROBI_TIME_LIMIT = 9      # gurobipy GRB.TIME_LIMIT


def lipmip_reference(network, domain, c_vector, primal_norm='linf', timeout=None,
                     num_threads=4, preact='naive_ia', verbose=False) -> dict:
    """Run LipMIP once and return {'exact', 'bound', 'incumbent', 'timed_out', 'gap', 'time'}.

    `exact` is the optimum when the solve finished and None on a timeout;
    `bound` is always the proved upper bound; `incumbent` the best point found
    (NaN if the solver stored none); `gap` Gurobi's relative MIP gap.
    """
    c = np.asarray(c_vector, dtype=float)
    problem = LipMIP(network, domain, c, primal_norm=primal_norm, preact=preact,
                     verbose=verbose, timeout=timeout, num_threads=num_threads)
    result = problem.compute_max_lipschitz()
    model = result.model
    timed_out = model.Status == GUROBI_TIME_LIMIT
    try:
        incumbent = float(model.ObjVal)
    except Exception:                                    # no feasible solution stored
        incumbent = math.nan
    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = math.nan
    bound = float(result.value)
    return {
        'exact': None if timed_out else bound,
        'bound': bound,
        'incumbent': incumbent,
        'timed_out': bool(timed_out),
        'status': int(model.Status),
        'gap': gap,
        'time': float(result.compute_time),
        'primal_norm': primal_norm,
        'timeout': timeout,
    }
