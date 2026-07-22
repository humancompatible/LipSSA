"""Baseline uniform-sampling convergence curves for the planted benchmark,
overlaid on the analytic miss-probability curve Pr[Lhat_n < c] = (1 - rho)^n.

Run (from the repo root, in the project's Python env):
    python scripts_testing/planted_uniform_baseline.py

Outputs PNGs + a markdown report in scripts_testing/planted_out/.
"""

from __future__ import annotations

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# make `import benchmarks` work when run as a plain script from the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks import utils                                        # noqa: E402
from benchmarks.planted import (                                    # noqa: E402
    PLANTED_FLAWED,
    PLANTED_MIN_GADGET,
    make_planted_net,
)

# --- Experiment configuration (named constants) ---------------------------
DTYPE = torch.float64
NORM_P = 2                          # experiment norm: induced ||.||_2
D = 64                              # full input width
DEPTH1 = 10                         # planted permutation-stack depth
DEPTH2 = 5                          # distractor depth
GEN_SEED = 20260719                 # network-generation seed (fixed instance)

K_GRID = [3, 5, 8]
C_GRID = [0.5, 1.0]
N_MULT = 10                         # sample budget n up to N_MULT * 2^k
N_REPEATS_PROB = 100               # sampler seeds for the probability curve (graph 2)
N_REPEATS_IQR = 20                # subset for the interquartile band (graph 1)
HIT_EPS = 1e-9                      # ||J|| >= c - HIT_EPS counts as "found"
BATCH = 512                        # jacobian batch size

# Distractor-gap calibration: how the distractor's plateau level depends on the
# slope range, measured on one (k, c) pair.
KNOB2_K, KNOB2_C = 5, 1.0
SLOPE_DISTS = [(0.2, 0.5), (0.2, 0.95), (0.5, 0.95)]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planted_out")


def running_max_norms(net, d, k, n, sampler_seed):
    """Uniform-sample x ~ U(-1,1)^d and return the running max of ||J(x_i)||_p,
    length n. This measures the induced operator norm of the full Jacobian."""
    g = torch.Generator().manual_seed(int(sampler_seed))
    out = torch.empty(n, dtype=DTYPE)
    filled = 0
    cur = torch.tensor(0.0, dtype=DTYPE)
    while filled < n:
        b = min(BATCH, n - filled)
        X = (torch.rand(b, d, generator=g, dtype=DTYPE) * 2 - 1)
        nv = utils.jacobian_norms(net, X, NORM_P)
        cm = torch.cummax(nv, dim=0).values
        cm = torch.maximum(cm, cur)
        out[filled:filled + b] = cm
        cur = cm[-1]
        filled += b
    return out.numpy()


def collect_curves(k, c, style=PLANTED_MIN_GADGET, slope_dist=(0.2, 0.95), repeats=N_REPEATS_PROB):
    """Return (n_axis, Lhat matrix [repeats x n]) for the given instance."""
    n = N_MULT * (2 ** k)
    net, meta = make_planted_net(d=D, k=k, depth1=DEPTH1, depth2=DEPTH2, c=c,
                                 seed=GEN_SEED, planted_style=style,
                                 slope_dist=slope_dist, dtype=DTYPE)
    mat = np.empty((repeats, n))
    for r in range(repeats):
        mat[r] = running_max_norms(net, D, k, n, sampler_seed=r)
    return np.arange(1, n + 1), mat, meta


def wilson_band(p_hat, n_rep, z=1.96):
    """Wilson score interval for a binomial proportion (graph-2 confidence band)."""
    denom = 1 + z ** 2 / n_rep
    centre = (p_hat + z ** 2 / (2 * n_rep)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n_rep + z ** 2 / (4 * n_rep ** 2))
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def plot_graph1(results, path):
    """L_hat_n / c vs n (log-x), median + interquartile band, all (k, c)."""
    plt.figure(figsize=(8, 5))
    for (k, c), (n_axis, mat) in results.items():
        sub = mat[:N_REPEATS_IQR] / c
        med = np.median(sub, axis=0)
        q1, q3 = np.percentile(sub, [25, 75], axis=0)
        line, = plt.plot(n_axis, med, label=f"k={k}, c={c}")
        plt.fill_between(n_axis, q1, q3, alpha=0.2, color=line.get_color())
    plt.axhline(1.0, ls="--", color="k", lw=0.8)
    plt.xscale("log")
    plt.xlabel("n (samples)"); plt.ylabel(r"$\hat{L}_n / c$")
    plt.title("Uniform sampling: normalised estimate vs budget (median + IQR)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_graph2(results, metas, path):
    """Empirical Pr[Lhat_n < c] vs n with analytic (1-rho)^n and Wilson band."""
    plt.figure(figsize=(8, 5))
    for (k, c), (n_axis, mat) in results.items():
        rho = metas[(k, c)].rho
        emp = np.mean(mat < c - HIT_EPS, axis=0)
        lo, hi = wilson_band(emp, mat.shape[0])
        analytic = (1 - rho) ** n_axis
        line, = plt.plot(n_axis, emp, label=f"k={k}, c={c} (emp)")
        plt.fill_between(n_axis, lo, hi, alpha=0.2, color=line.get_color())
        plt.plot(n_axis, analytic, ls="--", lw=1.0, color=line.get_color(),
                 label=f"k={k} analytic $(1-2^{{-k}})^n$")
    plt.xscale("log")
    plt.xlabel("n (samples)"); plt.ylabel(r"$\Pr[\hat{L}_n < c]$")
    plt.title("Uniform sampling: miss probability vs analytic curve")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_graph3(min_res, flawed_res, path):
    """min_gadget vs flawed_permutation miss-probability on (k=5, c=1.0)."""
    plt.figure(figsize=(8, 5))
    for label, (n_axis, mat, rho) in [("min_gadget", min_res), ("flawed_permutation", flawed_res)]:
        emp = np.mean(mat < 1.0 - HIT_EPS, axis=0)
        plt.plot(n_axis, emp, label=f"{label} (emp)")
        plt.plot(n_axis, (1 - rho) ** n_axis, ls="--", lw=1.0,
                 label=f"{label} analytic, rho={rho:.4f}")
    plt.xscale("log")
    plt.xlabel("n (samples)"); plt.ylabel(r"$\Pr[\hat{L}_n < c]$")
    plt.title("Measure inversion: min-gadget vs flawed permutation (k=5, c=1.0)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def distractor_plateau(k, c, slope_dist):
    """Empirical distractor plateau: median Lhat_n / c over samples taken BEFORE
    the planted orthant is hit (i.e. the level the estimate crawls at while it has
    only seen the distractor). Returns the median normalised plateau level."""
    n_axis, mat, meta = collect_curves(k, c, slope_dist=slope_dist, repeats=N_REPEATS_IQR)
    pre_hit = mat[mat < c - HIT_EPS] / c
    return float(np.median(pre_hit)) if pre_hit.size else float("nan"), meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()

    # --- Graphs 1 & 2: the (k, c) grid ---
    results, metas = {}, {}
    for k in K_GRID:
        for c in C_GRID:
            print(f"[grid] k={k} c={c} ...", flush=True)
            n_axis, mat, meta = collect_curves(k, c)
            results[(k, c)] = (n_axis, mat)
            metas[(k, c)] = meta

    plot_graph1(results, os.path.join(OUT_DIR, "graph1_normalised_estimate.png"))
    plot_graph2(results, metas, os.path.join(OUT_DIR, "graph2_miss_probability.png"))

    # --- Graph 3: min_gadget vs flawed_permutation on (k=5, c=1.0) ---
    print("[compare] min_gadget vs flawed_permutation (k=5, c=1.0) ...", flush=True)
    n_axis_m, mat_m, meta_m = collect_curves(5, 1.0, style=PLANTED_MIN_GADGET)
    n_axis_f, mat_f, meta_f = collect_curves(5, 1.0, style=PLANTED_FLAWED)
    plot_graph3((n_axis_m, mat_m, meta_m.rho), (n_axis_f, mat_f, meta_f.rho),
                os.path.join(OUT_DIR, "graph3_measure_inversion.png"))

    # --- Knob-2 calibration: distractor plateaus per slope_dist ---
    plateaus = {}
    for sd in SLOPE_DISTS:
        print(f"[knob2] slope_dist={sd} ...", flush=True)
        level, _ = distractor_plateau(KNOB2_K, KNOB2_C, sd)
        plateaus[sd] = level

    elapsed = time.time() - t0
    _write_report(results, metas, plateaus, elapsed)
    print(f"Done in {elapsed:.1f}s. Outputs in {OUT_DIR}")


def _write_report(results, metas, plateaus, elapsed):
    lines = []
    lines.append("# Planted benchmark - uniform-sampling baseline report\n")
    lines.append(f"- Norm: p = {NORM_P} (||J||_2); domain U(-1,1)^{D}; "
                 f"depth1={DEPTH1}, depth2={DEPTH2}, gen_seed={GEN_SEED}.")
    lines.append(f"- Repeats: {N_REPEATS_PROB} sampler seeds (prob curve), "
                 f"{N_REPEATS_IQR} (IQR band). Budget n up to {N_MULT}*2^k.")
    lines.append(f"- Total runtime: {elapsed:.1f}s.\n")

    lines.append("## Convergence to analytic curve (validation)\n")
    lines.append("| k | c | rho=2^-k | max |emp - (1-rho)^n| |")
    lines.append("|---|---|----------|------------------------|")
    for (k, c), (n_axis, mat) in results.items():
        rho = metas[(k, c)].rho
        emp = np.mean(mat < c - HIT_EPS, axis=0)
        max_dev = float(np.max(np.abs(emp - (1 - rho) ** n_axis)))
        lines.append(f"| {k} | {c} | {rho:.4g} | {max_dev:.3f} |")
    lines.append("\nIf the max deviation sits inside the Wilson band width "
                 "(~1/sqrt(100)=0.1), the generator and estimator are consistent "
                 "with the analytic prediction.\n")

    lines.append("## Distractor plateau levels vs slope_dist\n")
    lines.append(f"On (k={KNOB2_K}, c={KNOB2_C}). Plateau = median of Lhat_n/c over "
                 "pre-hit samples (the level the estimate crawls at before finding the orthant).\n")
    lines.append("| slope_dist | median plateau (Lhat/c) |")
    lines.append("|------------|-------------------------|")
    for sd, level in plateaus.items():
        lines.append(f"| {sd} | {level:.3f} |")
    lines.append("\nThese plateau levels quantify the gap Delta = c - plateau, "
                 "which controls how hard localising the maximum is.\n")

    lines.append("## flawed_permutation (measure inversion)\n")
    lines.append("Expected degenerate behaviour: Pr[Lhat_n < c] ~ 2^-k already at "
                 "n=1 (the reaching region is almost the whole domain). See "
                 "graph3_measure_inversion.png -- this is the intended behaviour of "
                 "the control construction, not a bug.\n")

    lines.append("## Open design questions\n")
    lines.append("- (a) Confirm the min-gadget as the primary planted construction.")
    lines.append("- (b) The planted block outputs a scalar (R^k -> R) rather than "
                 "R^k -> R^k -- a visible change in the block's output shape.")
    lines.append("- (c) Choice of the primary experiment norm p in {1, 2, inf} "
                 "(this run used p=2).")
    lines.append("- (d) slope_dist parameters for a larger follow-up grid "
                 "(see plateau table above).")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
