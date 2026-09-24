import math
import time

import numpy as np
import utilities as utils
import torch
from other_methods import OtherResult

_DUAL_NORM = {'linf': 1, 'l1': float('inf'), 'l2': 2}

class RegionNode:
    def __init__(self, lb, ub, maximum=0, minimum=float('inf'), mean=0, std=0, n=0,
                 device=torch.device('cpu')):
        self.lb = lb
        self.ub = ub
        self.n = n
        self.maximum = maximum
        self.minimum = minimum
        self.mean = mean
        self.std = std
        self.device = device
        self._lb_t = torch.as_tensor(self.lb, dtype=torch.float, device=self.device)
        self._ub_t = torch.as_tensor(self.ub, dtype=torch.float, device=self.device)
        self.left = None
        self.right = None
        self.mid = None  # [[lb', ub'] rows of get_middle] once this node is split

    def is_leaf(self):
        return self.left is None and self.right is None

    def add_evaluation(self, v):
        self.n += 1

        prev_mean = self.mean
        self.mean = self.mean + (v - self.mean) / self.n
        self.maximum = max(self.maximum, v)
        self.minimum = min(self.minimum, v)

        if self.n == 1:
            self.std = 0
        else:
            self.std = ((self.n - 1) * (self.std ** 2) + (v - prev_mean) * (v - self.mean)) / self.n
            self.std = math.sqrt(self.std)

    def get_random_points(self, n):
        """ `n` points uniform in the box, drawn on demand: shape [d] for n == 1,
        [n, d] otherwise. Returned as a leaf tensor with requires_grad set, ready
        for the Jacobian call.
        """
        return self.get_random_points_batch(n).squeeze(0).requires_grad_()

    def get_random_points_batch(self, n):
        return self._lb_t + torch.rand(
            n, len(self.lb), device=self.device
        ) * (self._ub_t - self._lb_t)

    def get_middle(self, axis=None):
        """ The box halved along `axis` (default: its longest side), as the
        [lb, ub] pair with that coordinate replaced by the midpoint.
        """
        d = (self.ub - self.lb).argmax() if axis is None else axis
        mid = np.array([self.lb, self.ub])
        m = (mid[0, d] + mid[1, d]) / 2
        mid[:, d] = m
        return mid


SPLIT_RULES = ('longest', 'max_var_axis', 'random')


class Space:
    def __init__(self, lb, ub, c, device=torch.device('cpu'), n0=10, split_rule='longest'):
        """ Binary tree of axis-aligned boxes over [lb, ub] with the UCB logic.

        ARGS:
            c: exploration coefficient in the UCB score
            n0: a leaf with at most n0 evaluations scores +inf, so every new
                leaf gets sampled n0 times before its statistics are trusted
            split_rule: which axis a box is halved along when it is split --
                'longest' (its longest side, ties broken uniformly), 'max_var_axis' (the axis whose
                midpoint split explains the most variance of the values seen
                in the box; falls back to 'longest' with fewer than two values
                on a side), or 'random' (an axis chosen uniformly)
        """
        assert split_rule in SPLIT_RULES, f"split_rule must be one of {SPLIT_RULES}"
        self.lb = lb
        self.ub = ub
        self.c = c
        self.n0 = n0
        self.split_rule = split_rule
        self.device = device
        self.capacity = 100
        self.eval_num = 0
        self.dimension = self.lb.shape[0]
        self.evaluations = np.zeros((self.capacity, self.dimension + 1), dtype=float) - np.inf
        self.root = RegionNode(self.lb, self.ub, device=self.device)

    def push_evaluation(self, v: RegionNode, x, fx):
        v.add_evaluation(fx)
        if v.is_leaf():
            return
        if (x <= v.mid[1]).all():
            self.push_evaluation(v.left, x, fx)
        else:
            self.push_evaluation(v.right, x, fx)

    def add_evaluation(self, x, fx):
        if self.eval_num >= self.capacity:
            self.capacity *= 2
            new_evals = np.zeros((self.capacity, self.dimension + 1), dtype=float) - np.inf
            new_evals[:self.eval_num] = self.evaluations
            self.evaluations = new_evals
        self.evaluations[self.eval_num] = np.append(x, fx)
        self.push_evaluation(self.root, x, fx)
        self.eval_num += 1

    def compute_ucb(self, v):
        if v.n <= self.n0:
            return np.inf
        bonus = math.sqrt(np.log(self.eval_num + 1) / v.n)
        return v.maximum + self.c * bonus * v.std

    def choose_region(self) -> RegionNode:
        leaves = self.get_leaves()
        ucb_vals = np.array([self.compute_ucb(leaf) for leaf in leaves])
        best = np.flatnonzero(ucb_vals == ucb_vals.max())
        return leaves[best[np.random.randint(len(best))]]

    def split_axis(self, node, X, fx):
        """ Axis to halve `node` along, per `self.split_rule`; `X`, `fx` are the
        evaluations that fall inside the node.
        """
        if self.split_rule == 'random':
            return np.random.randint(self.dimension)
        if self.split_rule == 'max_var_axis' and X.shape[0] >= 4:
            centre = (node.lb + node.ub) / 2
            left = X <= centre                            # [n, d] membership per axis
            n_l = left.sum(axis=0)
            n_r = X.shape[0] - n_l
            ok = (n_l >= 2) & (n_r >= 2)
            if ok.any():
                s_l = (left * fx[:, None]).sum(axis=0)
                m_l = np.where(ok, s_l / np.maximum(n_l, 1), 0.0)
                m_r = np.where(ok, (fx.sum() - s_l) / np.maximum(n_r, 1), 0.0)
                between = np.where(ok, n_l * n_r * (m_l - m_r) ** 2, -np.inf)
                return int(between.argmax())
        side = node.ub - node.lb
        longest = np.flatnonzero(side == side.max())
        return int(longest[np.random.randint(len(longest))])

    def increment(self):
        node = self.choose_region()

        X = self.evaluations[:, :self.dimension]
        fx = self.evaluations[:, -1]
        inside = ((X >= node.lb) & (X <= node.ub)).all(axis=1)
        mid = node.get_middle(self.split_axis(node, X[inside], fx[inside]))
        node.mid = mid

        mask = ((X >= node.lb) & (X <= mid[1])).all(axis=1)
        evals = fx[mask]
        n_maximum = 0.0
        n_minimum = float('inf')
        n_mean = 0.0
        n_std = 0.0
        if evals.shape[0] > 0:
            n_maximum = np.max(evals)
            n_minimum = np.min(evals)
            n_mean = np.mean(evals)
            n_std = np.std(evals)
        node.left = RegionNode(lb=node.lb, ub=mid[1], maximum=n_maximum, minimum=n_minimum, mean=n_mean, std=n_std, n=evals.shape[0], device=self.device)

        mask = ((X >= mid[0]) & (X <= node.ub)).all(axis=1)
        evals = fx[mask]
        n_maximum = 0.0
        n_minimum = float('inf')
        n_mean = 0.0
        n_std = 0.0
        if evals.shape[0] > 0:
            n_maximum = np.max(evals)
            n_minimum = np.min(evals)
            n_mean = np.mean(evals)
            n_std = np.std(evals)
        node.right = RegionNode(lb=mid[0], ub=node.ub, maximum=n_maximum, minimum=n_minimum, mean=n_mean, std=n_std, n=evals.shape[0], device=self.device)

    def get_leaves(self, v=None) -> list:
        if v is None:
            v = self.root
        if v.is_leaf():
            return [v]
        return self.get_leaves(v.left) + self.get_leaves(v.right)


class StochasticApproximationUCBDynamic(OtherResult):
    def __init__(self, network, c_vector, domain, c, partition_step, primal_norm='linf', device='cpu',
                 is_transformer=False, n0=10, split_rule='longest'):
        """ UCB bandit over an adaptive binary partition of `domain`.

        ARGS:
            c: exploration coefficient (higher = more exploration)
            partition_step: the tree is split at the first iteration >= each of
                partition_step, partition_step**2, partition_step**3, ...
                (a non-integer step such as 1.5 gives a denser schedule)
            n0, split_rule: see Space
        """
        super(StochasticApproximationUCBDynamic, self).__init__(network, c_vector, domain, primal_norm)
        assert utils.arraylike(c_vector)
        self.DEVICE = torch.device(device)
        if not isinstance(self.c_vector, torch.Tensor):
            self.c_vector = torch.tensor(self.c_vector, dtype=torch.float)
        self.c_vector = self.c_vector.to(self.DEVICE)
        self.network = self.network.to(self.DEVICE)
        self.value = torch.tensor([1e-18]).to(self.DEVICE)
        self.answer_coords = None
        self.iteration_count = 0
        self.lb = domain.box_low.cpu().detach().numpy()
        self.ub = domain.box_hi.cpu().detach().numpy()
        self.c = c
        self.partition_step = partition_step
        self.side = self.ub - self.lb
        self.space = Space(self.lb, self.ub, self.c, device=self.DEVICE, n0=n0, split_rule=split_rule)
        self.is_transformer = is_transformer

    def f(self, point):
        dual_p = _DUAL_NORM[self.primal_norm]
        if self.is_transformer:
            j_norm = torch.autograd.functional.jacobian(
                lambda point: self.network(point).squeeze(0).mv(self.c_vector).sum(), point
            ).norm(p=dual_p)
        else:
            j_norm = torch.autograd.functional.jacobian(
                lambda point: self.network(point).mv(self.c_vector).sum(), point
            ).norm(p=dual_p)
        return j_norm

    def compute(self, max_iter=1000, v=False, exact=None, tol=1e-5, mode="Absolute"):
        """ UCB search for max_iter iterations (one point per iteration).

        `self.history` records every improvement of the running maximum as
        [iteration, seconds since start, value]; the best-so-far curve is a step
        function, so this is its exact and compact description.
        """
        timer = utils.Timer()
        t0 = time.perf_counter()
        self.iteration_count = 0
        self.history = []
        next_partition = self.partition_step
        step_mul = self.partition_step

        for it in range(max_iter):
            if it >= next_partition:
                self.space.increment()
                next_partition = next_partition * step_mul

            reg = self.space.choose_region()
            x = reg.get_random_points(1)
            if self.is_transformer:
                x = x.expand(1,1,64)
            fx = self.f(x)
            fx_scalar = float(fx.detach().cpu().item())
            x_np = x.detach().cpu().numpy().reshape(-1)[:self.space.dimension]
            self.space.add_evaluation(x_np, fx_scalar)
            self.iteration_count += 1

            if self.value < fx:
                self.value = torch.maximum(self.value, fx)
                self.answer_coords = x.detach().cpu().numpy()
                self.history.append([self.iteration_count, time.perf_counter() - t0, fx_scalar])

            if v:
                print(f"Current approximate: {self.value:.4f}")
            if exact is not None:
                if mode == "Absolute":
                    if torch.abs(exact - self.value) <= tol:
                        break
                else:
                    if torch.abs(exact - self.value) / self.value * 100.0 <= tol:
                        break

        self.compute_time = timer.stop()
        return self.value


if __name__ == '__main__':
    from hyperbox import Hyperbox

    def walk_tree(v):
        print(f"{v.lb, v.ub, v.mean, v.std, v.minimum, v.maximum}")
        if v.left is not None:
            walk_tree(v.left)
        if v.right is not None:
            walk_tree(v.right)

    DIMENSION = 3
    domain = Hyperbox.build_custom_hypercube(DIMENSION, 5, 5.0)
    lb = domain.box_low.cpu().detach().numpy()
    ub = domain.box_hi.cpu().detach().numpy()
    r = RegionNode(lb, ub)
    print(f"mid =\n {r.get_middle()}\n")
    space = Space(lb, ub, c=1.0)
    space.increment()
    space.increment()
    space.increment()
    walk_tree(space.root)
    print("\n")
    x = np.array([[1.4, 2, 7], [0, 0, 0], [7, 7, 7]])
    v = [1.0, 1.4, 2.5]
    a = []
    i = 0
    for X in x:
        space.push_evaluation(space.root, X, v[i])
        a.append(v[i])
        i += 1
    print(f"means: np = {np.mean(np.array(a))}, est = {space.root.mean}")
    print(f"stds: np = {np.std(np.array(a))}, est = {space.root.std}\n\n")

    walk_tree(space.root)