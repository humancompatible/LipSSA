"""Binary-MNIST comparison table: LipMIP, LipSDP and the two sampling estimators.

Networks are trained on 1 vs 7 (seed 0) and evaluated on the whole input cube
[0,1]^784 with c = [1, -1] and primal norm linf, so every method reports a value
of max ||grad <c, f>||_1:

  * LipMIP: exact when it finishes; on a timeout its proved bound and the best
    point it found are both kept, and the row reports the interval between the
    best value found by any method and the best bound proved by any method.
  * LipSDP: bounds the global l2 constant; ||g||_1 <= sqrt(n) ||g||_2 turns it
    into an upper bound in this norm.
  * Algorithms 1 and 3: mean +- std over seeds at a fixed evaluation budget.

Every cell is cached per (network weights, method, seed), keyed by a fingerprint
of the weights, so methods run in different jobs are compared on the same network
and an interrupted run resumes.

Workflow:
    python scripts_testing/mnist_table.py --prepare       # laptop: train, save weights, LipSDP
    python scripts_testing/mnist_table.py --task T         # cluster: T in 0..2*len(ARCHS)-1
    python scripts_testing/mnist_table.py --render-only   # print the table from the cache
    python scripts_testing/mnist_table.py --smoke         # two small nets, everything, minutes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import cache, nets, runners
from experiments.ground_truth import lipmip_reference
from hyperbox import Hyperbox
from relu_nets import ReLUNet

EXPERIMENT = 'mnist_table'
ARCHS = [[784, 8, 8, 2],
         [784, 20, 20, 2],
         [784, 8, 8, 8, 2],
         [784, 8, 8, 8, 8, 8, 2],
         [784, 20, 20, 20, 20, 20, 2]]
SMOKE_ARCHS = [[784, 8, 2], [784, 8, 8, 2]]
DIGITS = (1, 7)
TRAIN_SEED = 0
PRIMAL_NORM = 'linf'
METHODS = ('uniform', 'ucb')
UCB_C = 5
UCB_STEP = 2
UCB_SPLIT_TIES = 'uniform'     # cache tag: splits break ties between longest sides uniformly (not the first one)
LIPMIP_TIMEOUT = 10_800        # seconds
BUDGET = 500_000              # evaluations per sampling run
N_SEEDS = 5
UCB_RNG_OFFSET = 1000          # Algorithm 3 draws from a random stream independent of Algorithm 1's
WEIGHTS_DIR = cache.DEFAULT_ROOT / EXPERIMENT / 'weights'


def network(arch, allow_train):
    """The trained network for `arch`, loaded from WEIGHTS_DIR if saved there,
    otherwise trained with TRAIN_SEED and saved (only when `allow_train`)."""
    path = WEIGHTS_DIR / f"{'-'.join(map(str, arch))}_seed{TRAIN_SEED}.pt"
    if path.exists():
        net = ReLUNet(arch)
        net.load_state_dict(torch.load(path))
        net.eval()
    elif allow_train:
        net, _, _ = nets.train_mnist(arch, seed=TRAIN_SEED, digits=DIGITS)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
    else:
        raise FileNotFoundError(f"{path} is missing; weights are trained with --prepare and copied over")
    return net, Hyperbox.build_unit_hypercube(784), torch.tensor([1.0, -1.0])


def fingerprint(net):
    h = hashlib.sha1()
    for p in net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def lipsdp_reference(net, c_vector):
    """LipSDP-Neuron's global l2 bound and its sqrt(n) conversion to this norm."""
    from other_methods import LipSDP
    if LipSDP is None:
        raise RuntimeError("LipSDP needs the MATLAB engine")
    sdp = LipSDP(net, c_vector)
    l2 = float(sdp.compute())
    return {'l2': l2, 'bound': l2 * float(np.sqrt(net.layer_sizes[0])),
            'time': float(sdp.compute_time)}


def cell(compute, key, fn, force):
    """Compute-and-cache the cell, or only read it (None if absent).
    A cell that is actually computed (not found in the cache) prints one line."""
    if not compute:
        return cache.load(cache.result_path(EXPERIMENT, key))
    fresh = force or cache.load(cache.result_path(EXPERIMENT, key)) is None
    record = cache.run_cached(EXPERIMENT, key, fn, force=force)
    if fresh:
        what = key.get('method') or key.get('solver')
        seed = f" seed {key['seed']}" if 'seed' in key else ''
        value = record['value'] if 'value' in record else record['bound']
        seconds = record['compute_time'] if 'compute_time' in record else record['time']
        print(f"  {key['arch']} {what}{seed}: {value:.5f}  ({seconds:.1f} s)", flush=True)
    return record


def row(arch, timeout, budget, n_seeds, parts, allow_train, force=False):
    """One table row: LipMIP, LipSDP and the samplers on one network.

    `parts` is the subset of {'lipmip', 'lipsdp', 'sampling'} to compute; all
    other cells are only read from the cache, and missing ones come back None.
    """
    net, domain, cv = network(arch, allow_train)
    base = dict(arch=arch, digits=list(DIGITS), train_seed=TRAIN_SEED,
                weights=fingerprint(net), primal_norm=PRIMAL_NORM)

    gt = cell('lipmip' in parts, dict(base, solver='lipmip', timeout=timeout),
              lambda: lipmip_reference(net, domain, cv, primal_norm=PRIMAL_NORM, timeout=timeout),
              force)
    sdp = cell('lipsdp' in parts, dict(base, solver='lipsdp'),
               lambda: lipsdp_reference(net, cv), force)

    samples = {}
    for method in METHODS:
        params = dict(c=UCB_C, partition_step=UCB_STEP) if method == 'ucb' else {}
        offset = UCB_RNG_OFFSET if method == 'ucb' else 0
        tag = dict(split_ties=UCB_SPLIT_TIES) if method == 'ucb' else {}
        records = [cell('sampling' in parts, dict(base, method=method, seed=s, budget=budget, **params, **tag),
                        lambda s=s: runners.run_method(method, net, cv, domain, budget, s + offset,
                                                       primal_norm=PRIMAL_NORM, **params),
                        force)
                   for s in range(n_seeds)]
        samples[method] = None if any(r is None for r in records) else records
    return gt, sdp, samples


def render(rows, budget, n_seeds):
    print(f"\nBinary MNIST {DIGITS[0]} vs {DIGITS[1]}, domain [0,1]^784, c = [1, -1], "
          f"primal norm {PRIMAL_NORM} (objective ||grad||_1)")
    print(f"Algorithms 1 and 3: {n_seeds} seeds, {budget:,} evaluations each, mean +- std; "
          f"Algorithm 3 with c = {UCB_C}, partition step {UCB_STEP}.\n")
    head = f"{'network':<26} {'method':<14} {'value':>24} {'time (s)':>10}  notes"
    print(head)
    print('-' * len(head))
    for arch, (gt, sdp, samples) in rows:
        ref = None if gt is None else gt['exact']
        if gt is None:
            print(f"{str(arch):<26} {'LipMIP':<14} {'(not computed)':>24}")
        elif ref is not None:
            print(f"{str(arch):<26} {'LipMIP':<14} {ref:>24.5f} {gt['time']:>10.1f}  exact")
        else:
            print(f"{str(arch):<26} {'LipMIP':<14} {'<= %.5f' % gt['bound']:>24} {gt['time']:>10.1f}  "
                  f"timeout; gap {gt['gap']:.1%}; found {gt['incumbent']:.5f}")

        if sdp is None:
            print(f"{'':<26} {'LipSDP':<14} {'(not computed)':>24}")
        else:
            note = f"l2 bound {sdp['l2']:.5f} x sqrt(784)"
            if ref is not None:
                note += f"; {(sdp['bound'] - ref) / ref:+.2%}"
            print(f"{'':<26} {'LipSDP':<14} {'<= %.5f' % sdp['bound']:>24} {sdp['time']:>10.1f}  {note}")

        best_found = 0.0
        for method, records in samples.items():
            label = {'uniform': 'Algorithm 1', 'ucb': 'Algorithm 3'}[method]
            if records is None:
                print(f"{'':<26} {label:<14} {'(not computed)':>24}")
                continue
            vals = np.array([r['value'] for r in records])
            times = np.array([r['compute_time'] for r in records])
            best_found = max(best_found, vals.max())
            note = '' if ref is None else f"{(vals.mean() - ref) / ref:+.2%}"
            print(f"{'':<26} {label:<14} {vals.mean():>14.5f} +- {vals.std():<7.5f} {times.mean():>10.1f}  {note}")

        if gt is not None and ref is None:
            lower = max(best_found, gt['incumbent'] if np.isfinite(gt['incumbent']) else 0.0)
            upper = min([gt['bound']] + ([sdp['bound']] if sdp is not None else []))
            print(f"{'':<26} {'true value in':<14} {'[%.5f, %.5f]' % (lower, upper):>24} "
                  f"{'':>10}  best found / best proved")
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--timeout', type=int, default=LIPMIP_TIMEOUT, help='LipMIP time limit, seconds')
    p.add_argument('--budget', type=int, default=BUDGET, help='evaluations per sampling run')
    p.add_argument('--seeds', type=int, default=N_SEEDS, help='sampling seeds per network')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true',
                      help='train and save every network, compute LipSDP (needs MATLAB)')
    mode.add_argument('--task', type=int, default=None,
                      help='one cluster task: T < len(ARCHS) runs LipMIP on ARCHS[T], '
                           'otherwise the sampling runs on ARCHS[T - len(ARCHS)]')
    mode.add_argument('--render-only', action='store_true', help='print the table from the cache')
    mode.add_argument('--smoke', action='store_true', help='two small nets, every method, 60 s, 5k evaluations')
    p.add_argument('--force', action='store_true', help='recompute cached cells')
    args = p.parse_args(argv)

    archs, timeout, budget, seeds = ARCHS, args.timeout, args.budget, args.seeds
    parts, allow_train = set(), False
    if args.prepare:
        parts, allow_train = {'lipsdp'}, True
    elif args.task is not None:
        archs = [ARCHS[args.task % len(ARCHS)]]
        parts = {'lipmip'} if args.task < len(ARCHS) else {'sampling'}
    elif args.smoke:
        archs, timeout, budget, seeds = SMOKE_ARCHS, 60, 5_000, 2
        parts, allow_train = {'lipmip', 'lipsdp', 'sampling'}, True

    rows = []
    for arch in archs:
        if parts:
            print(f"=== {arch}: {', '.join(sorted(parts))} ===", flush=True)
        rows.append((arch, row(arch, timeout, budget, seeds, parts, allow_train, force=args.force)))
    render(rows, budget, seeds)


if __name__ == '__main__':
    main()
