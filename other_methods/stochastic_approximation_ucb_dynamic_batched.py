"""
Batched variant of StochasticApproximationUCBDynamic.

Same UCB-bandit-over-binary-tree algorithm as
``stochastic_approximation_ucb_dynamic.py``, but each "step" evaluates a batch
of ``batch_size`` points from the chosen region in a single forward+backward
pass. This is the variant that actually benefits from a GPU.

Per-sample gradient trick (no functorch dependency)
---------------------------------------------------
For the standard MLP / ReLU case (no cross-sample interaction in the network),
the Jacobian of the batched scalar output w.r.t. the batched input is
block-diagonal. So a single backward of ``y.sum()`` returns a tensor of shape
``[B, d]`` whose i-th row is the per-sample gradient ``d y_i / d x_i``. Taking
``norm(p=dual_p, dim=-1)`` then gives the B per-sample dual-norm gradient
norms in one backward. This is what makes the batch GPU-efficient.

Caveats
-------
* Requires the network to be applied to a batched input ``[B, d]`` and produce
  a batched output ``[B, out]``. ReLUNet/Sequential MLPs satisfy this.
* If the network has cross-sample interaction (e.g. BatchNorm in train mode),
  the trick is unsound -- use ``torch.func.vmap(jacrev(...))`` instead.
* Transformer mode (``is_transformer=True``) is intentionally not supported in
  this batched variant; use the unbatched class for that.
* The partition schedule is in number of *batched steps*, identical to the
  unbatched version's iteration count. Total point evaluations is
  ``max_iter * batch_size``.

Usage
-----
    solver = StochasticApproximationUCBDynamicBatched(
        network, c_vector, domain,
        c=15, partition_step=2, batch_size=256,
        primal_norm='linf', device='cuda',
    )
    value = solver.compute(max_iter=500)   # 500 * 256 = 128k point evaluations
"""

import math
import time
import numpy as np
import torch

import utilities as utils
from other_methods import OtherResult
from other_methods.stochastic_approximation_ucb_dynamic import Space


_DUAL_NORM = {'linf': 1, 'l1': float('inf'), 'l2': 2}


class StochasticApproximationUCBDynamicBatched(OtherResult):
    def __init__(self, network, c_vector, domain, c, partition_step,
                 batch_size=256, primal_norm='linf', device='cpu'):
        super().__init__(network, c_vector, domain, primal_norm)
        assert utils.arraylike(c_vector)
        assert batch_size >= 1
        self.DEVICE = torch.device(device)
        if not isinstance(self.c_vector, torch.Tensor):
            self.c_vector = torch.tensor(self.c_vector, dtype=torch.float)
        self.c_vector = self.c_vector.to(self.DEVICE)
        self.network = self.network.to(self.DEVICE)
        self.value = torch.tensor([1e-18], device=self.DEVICE)
        self.answer_coords = None
        self.iteration_count = 0
        self.batch_size = batch_size
        self.lb = domain.box_low.cpu().detach().numpy()
        self.ub = domain.box_hi.cpu().detach().numpy()
        self.c = c
        self.partition_step = partition_step
        self.space = Space(self.lb, self.ub, self.c, device=self.DEVICE)

    def f_batch(self, x_batch):
        """ Computes the per-sample dual-norm of the gradient for a batch.
        ARGS:
            x_batch: [B, d] tensor on self.DEVICE, requires_grad=True
        RETURNS:
            [B] tensor of dual-norm-of-gradient values (no grad)
        """
        dual_p = _DUAL_NORM[self.primal_norm]
        # Scalar output: sum of <c, f(x_i)> over the batch. Because the
        # network has no cross-sample interaction, d/d x_i of this sum is the
        # per-sample gradient d/d x_i of <c, f(x_i)>.
        out = self.network(x_batch)              # [B, out_dim]
        scalar = out.matmul(self.c_vector).sum() # ()
        grads, = torch.autograd.grad(scalar, x_batch, create_graph=False)
        # grads is [B, d]; take dual norm along d
        return grads.detach().norm(p=dual_p, dim=-1)

    def compute(self, max_iter=1000, v=False, exact=None, tol=1e-5, mode="Absolute", max_seconds=None):
        """ UCB search for max_iter batched steps, or until max_seconds of wall
        time have passed. `self.history` records every improvement of the
        running maximum as [point evaluations, seconds since start, value].
        """
        timer = utils.Timer()
        t0 = time.perf_counter()
        self.iteration_count = 0
        self.history = []
        next_partition = self.partition_step
        step_mul = self.partition_step

        for it in range(max_iter):
            if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
                break
            if it >= next_partition:
                self.space.increment()
                next_partition = next_partition * step_mul

            reg = self.space.choose_region()
            x_batch = reg.get_random_points_batch(self.batch_size)
            x_batch = x_batch.detach().requires_grad_(True)

            fx_batch = self.f_batch(x_batch)        # [B] on device

            # Update best
            best_idx = int(torch.argmax(fx_batch).item())
            best_fx = fx_batch[best_idx]
            if self.value < best_fx:
                self.value = torch.maximum(self.value, best_fx.unsqueeze(0))
                self.answer_coords = x_batch[best_idx].detach().cpu().numpy()
                self.history.append([(it + 1) * self.batch_size, time.perf_counter() - t0, float(best_fx)])

            # Push all B evaluations into the tree (CPU bookkeeping)
            x_np = x_batch.detach().cpu().numpy()
            fx_np = fx_batch.detach().cpu().numpy()
            for i in range(self.batch_size):
                self.space.add_evaluation(x_np[i], float(fx_np[i]))

            self.iteration_count += 1

            if v:
                cur = float(self.value.detach().cpu().item())
                print(f"step {it}  best={cur:.4f}  batch_max={float(best_fx):.4f}")

            if exact is not None:
                cur = float(self.value.detach().cpu().item())
                if mode == "Absolute":
                    if abs(float(exact) - cur) <= tol:
                        break
                else:
                    if abs(float(exact) - cur) / max(cur, 1e-12) * 100.0 <= tol:
                        break

        self.compute_time = timer.stop()
        return self.value
