import numpy as np
import utilities as utils
import torch
from .other_methods import OtherResult

_DUAL_NORM = {'linf': 1, 'l1': float('inf'), 'l2': 2}


class StochasticApproximation(OtherResult):

    def __init__(self, network, c_vector, domain, primal_norm='linf', device='cpu'):
        super(StochasticApproximation, self).__init__(network, c_vector, domain, primal_norm)
        assert utils.arraylike(c_vector)
        self.DEVICE = torch.device(device)
        if not isinstance(self.c_vector, torch.Tensor):
            self.c_vector = torch.tensor(self.c_vector, dtype=torch.float)
        self.c_vector = self.c_vector.to(self.DEVICE)
        self.value = torch.tensor([1e-18]).to(self.DEVICE)
        self.iteration_count = 0
        self.network = self.network.to(self.DEVICE)
        self.eval_list = []

    def f(self, point):
        dual_p = _DUAL_NORM[self.primal_norm]
        j_norm = torch.autograd.functional.jacobian(
            lambda point: self.network(point).mv(self.c_vector).sum(), point
        ).norm(p=dual_p)
        return j_norm

    def compute(self, max_iter=10000, track_evaluations=False, v=False, exact=None, tol=1e-5, mode="Absolute"):
        timer = utils.Timer()
        self.iteration_count = 0
        random_pts = self.domain.random_point(num_points=max_iter, requires_grad=False)
        random_pts = random_pts.to(self.DEVICE).detach().requires_grad_(True)

        for it in range(max_iter):
            point = random_pts[it]
            # nt_out = self.network(point)
            # gr = torch.autograd.grad(inputs=point, outputs=nt_out)[0].detach().norm(p=1)
            # self.value = torch.maximum(self.value, gr)
            # if track_evaluations:
            #     self.eval_list.append(gr.detach().cpu().numpy())
            # self.iteration_count += 1

            fx = self.f(point)
            if self.value < fx:
                self.value = torch.maximum(self.value, fx)
            self.iteration_count += 1

            if v:
                print(f"Current approximate: {self.value:.4f}")
            if exact is not None:
                if mode == "Absolute":
                    if torch.abs(exact - self.value) <= tol:
                        break
                else:
                    if torch.abs(exact - self.value)/self.value*100.0 <= tol:
                        break
        self.compute_time = timer.stop()
        return self.value.cpu().detach().numpy().squeeze()
