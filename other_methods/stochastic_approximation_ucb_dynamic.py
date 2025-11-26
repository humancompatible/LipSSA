import numpy as np
import math
import utilities as utils
import torch
from other_methods import OtherResult


class RegionNode:
    def __init__(self, lb, ub, maximum=0, mean=0, std=0, n=0):
        self.lb = lb
        self.ub = ub
        self.n = n
        self.maximum = maximum
        self.mean = mean
        self.std = std
        self.precomputed_random = torch.tensor(self.lb) + torch.rand(10005, len(self.lb)) * torch.tensor(self.ub - self.lb)
        self.random_idx = 0
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def add_evaluation(self, v):
        self.n += 1

        prev_mean = self.mean
        self.mean = self.mean + (v - self.mean) / self.n
        self.maximum = max(self.maximum, v)

        if self.n == 1:
            self.std = 0
        else:
            self.std = ((self.n - 1) * (self.std ** 2) + (v - prev_mean) * (v - self.mean)) / self.n
            self.std = math.sqrt(self.std)

    def get_random_points(self, n):
        # p = np.array([np.random.uniform(self.lb[i], self.ub[i], 1)[0] for i in range(len(self.lb))])
        # p = torch.tensor(p, dtype=torch.float, requires_grad=True)
        p = self.precomputed_random[self.random_idx]
        self.random_idx += 1
        if self.random_idx == 10000:
            self.random_idx = 0
            self.precomputed_random = torch.tensor(self.lb) + torch.rand(10005, len(self.lb)) * torch.tensor(
                self.ub - self.lb)
        return p.clone().detach().requires_grad_()

    def get_middle(self):
        d = (self.ub - self.lb).argmax()
        mid = np.array([self.lb, self.ub])
        m = (mid[0, d] + mid[1, d]) / 2
        mid[:, d] = m
        return mid


class Space:
    def __init__(self, lb, ub, c):
        self.lb = lb
        self.ub = ub
        self.c = c
        self.capacity = 100
        self.eval_num = 0
        self.dimension = self.lb.shape[0]
        self.evaluations = np.zeros((self.capacity, self.dimension + 1), dtype=float) - np.inf
        self.root = RegionNode(self.lb, self.ub)

    def push_evaluation(self, v: RegionNode, x, fx):
        v.add_evaluation(fx)
        if v.is_leaf():
            return
        mid = v.get_middle()[1]
        if (x <= mid).all():
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

    # def compute_ucb(self, v: RegionNode, root_std):
    #     if v.n <= 10:
    #         return np.inf
    #     eps = 1e-10
    #     return v.maximum + self.c * math.sqrt(np.log(self.eval_num + 1) / v.n) * (v.std / (root_std + eps))

    def compute_ucb_alt(self, v):
        if v.n <= 10:
            return np.inf
        return v.maximum + self.c * math.sqrt(np.log(self.eval_num + 1) / v.n) * v.std

    def choose_region(self) -> RegionNode:
        leaves = self.get_leaves()
        ucb_vals = np.array([self.compute_ucb_alt(leaf) for leaf in leaves])
        idx = np.argmax(ucb_vals)

        return leaves[idx]


    def increment(self):
        node = self.choose_region()
        mid = node.get_middle()

        X = self.evaluations[:, :self.dimension]
        fx = self.evaluations[:, -1]

        mask = ((X >= node.lb) & (X <= mid[1])).all(axis=1)
        evals = fx[mask]
        n_maximum = 0.0
        n_mean = 0.0
        n_std = 0.0
        if evals.shape[0] > 0:
            n_maximum = np.max(evals)
            n_mean = np.mean(evals)
            n_std = np.std(evals)
        node.left = RegionNode(lb=node.lb, ub=node.get_middle()[1], maximum=n_maximum, mean=n_mean, std=n_std, n=evals.shape[0])

        mask = ((X >= mid[0]) & (X <= node.ub)).all(axis=1)
        evals = fx[mask]
        n_maximum = 0.0
        n_mean = 0.0
        n_std = 0.0
        if evals.shape[0] > 0:
            n_maximum = np.max(evals)
            n_mean = np.mean(evals)
            n_std = np.std(evals)
        node.right = RegionNode(lb=node.get_middle()[0], ub=node.ub, maximum=n_maximum, mean=n_mean, std=n_std, n=evals.shape[0])

    def get_leaves(self, v=None) -> list:
        if v is None:
            v = self.root
        if v.is_leaf():
            return [v]
        return self.get_leaves(v.left) + self.get_leaves(v.right)


class StochasticApproximationUCBDynamic(OtherResult):
    def __init__(self, network, c_vector, domain, c, partition_step, primal_norm='linf', device='cpu', use_c_vector=False, is_transformer=False):
        super(StochasticApproximationUCBDynamic, self).__init__(network, c_vector, domain, primal_norm)
        self.DEVICE = torch.device(device)
        self.network = self.network.to(self.DEVICE)
        self.value = torch.tensor([1e-18]).to(device)
        self.answer_coords = None
        self.iteration_count = 0
        self.lb = domain.box_low.cpu().detach().numpy()
        self.ub = domain.box_hi.cpu().detach().numpy()
        self.c = c
        self.partition_step = partition_step
        self.side = self.ub - self.lb
        self.space = Space(self.lb, self.ub, self.c)
        self.use_c_vector = use_c_vector
        self.is_transformer = is_transformer

    def f(self, point):
        if not self.use_c_vector:
            nt_out = self.network(point)
            grad_vectors = []
            for i in range(nt_out.shape[1]):
                grad_vectors.append(torch.autograd.grad(nt_out[0, i], point, retain_graph=True)[0])
            J = torch.stack(grad_vectors, dim=0)
            j_norm = J.norm(p=1)
            return j_norm
        else:
            if not self.is_transformer:
                j_norm = torch.autograd.functional.jacobian(lambda point: self.network(point).mv(self.c_vector).sum(),point).norm(p=1)
            else:
                # print(point.shape)
                # print(self.network(point).shape, self.c_vector)
                j_norm = torch.autograd.functional.jacobian(lambda point: self.network(point).squeeze(0).mv(self.c_vector).sum(),point).norm(p=1)
            return j_norm

    def compute(self, max_iter=1000, v=False, exact=None, tol=1e-5, mode="Absolute"):
        timer = utils.Timer()
        self.iteration_count = 0
        next_partition = self.partition_step
        step_mul = self.partition_step

        for it in range(max_iter):
            if it == next_partition:
                self.space.increment()
                next_partition = next_partition * step_mul

            reg = self.space.choose_region()
            x = reg.get_random_points(1)
            if self.is_transformer:
                x = x.expand(1,1,64)
            fx = self.f(x)
            self.space.add_evaluation(x.cpu().detach().numpy(), fx)

            if self.value < fx:
                self.value = torch.maximum(self.value, fx)
                self.answer_coords = x.detach().cpu().numpy()
            self.iteration_count += 1

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
        print(f"{v.lb, v.ub, v.mean, v.std}")
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
