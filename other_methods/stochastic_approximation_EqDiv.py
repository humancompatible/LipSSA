import numpy as np
import utilities as utils
import torch
from .other_methods import OtherResult


class Region:
    def __init__(self, l, r):
        self.lb = l
        self.ub = r
        self.mean = 0
        self.std = None
        self.maximum = 0
        self.sum_sq = 0
        self.n = 0

    def add_evaluation(self, v):
        self.n += 1

        self.maximum = np.maximum(self.maximum, v)

        prev_mean = self.mean
        self.mean = self.mean + (v - self.mean) / self.n

        self.sum_sq = self.sum_sq + (v - prev_mean) * (v - self.mean)
        self.std = (self.sum_sq / self.n) ** 0.5

    def get_random_point(self):
        p = np.array([np.random.uniform(self.lb[i], self.ub[i], 1)[0] for i in range(len(self.lb))])
        p = torch.tensor(p, dtype=torch.float, requires_grad=True)
        return p


class StochasticApproximationEqDiv(OtherResult):
    def __init__(self, network, c_vector, domain, divisions_per_dimension, primal_norm='linf', device='cpu', use_c_vector=False):
        super(StochasticApproximationEqDiv, self).__init__(network, c_vector, domain, primal_norm)
        self.value = 1e-18
        self.answer_coords = None
        self.iteration_count = 0
        self.num_divisions = divisions_per_dimension
        self.lb = domain.box_low.cpu().detach().numpy()
        self.ub = domain.box_hi.cpu().detach().numpy()
        self.regions = None
        self.side = self.ub - self.lb
        self.DEVICE = torch.device(device)
        self.network = self.network.to(self.DEVICE)
        self.use_c_vector = use_c_vector

        self.eval_list = []

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
            j_norm = torch.autograd.functional.jacobian(lambda point: self.network(point).mv(self.c_vector).sum(),
                                                        point).norm(p=1)
            return j_norm

    def init_regions(self):
        self.regions = []
        step = self.side / self.num_divisions
        dimensions = self.domain.dimension
        ll_points = np.array(np.meshgrid(*[np.arange(self.num_divisions) for _ in range(dimensions)])).T.reshape(-1, dimensions) * step
        ll_points = ll_points + self.lb
        rr_points = ll_points + step
        for i in range(ll_points.shape[0]):
            region = Region(ll_points[i], rr_points[i])
            self.regions.append(region)

    def compute(self, total_iterations=1000, track_evaluations=False, v=False, exact=None, tol=1e-5, mode="Absolute"):
        timer = utils.Timer()
        self.init_regions()
        iter_per_reg = total_iterations // len(self.regions)
        self.iteration_count = 0

        for region in self.regions:
            for _ in range(iter_per_reg):
                x_next = region.get_random_point()
                y_next = self.f(x_next)
                if track_evaluations:
                    region.add_evaluation(y_next)
                    self.eval_list.append(x_next.detach().cpu().numpy())
                self.iteration_count += 1
                if self.value < y_next:
                    self.value = np.maximum(self.value, y_next)
                    self.answer_coords = x_next.detach().cpu().numpy()
            if exact is not None:
                if mode == "Absolute":
                    if torch.abs(exact - self.value) <= tol:
                        break
                else:
                    if torch.abs(exact - self.value) / self.value * 100.0 <= tol:
                        break

        self.compute_time = timer.stop()
        return self.value
