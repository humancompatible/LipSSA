import numpy as np
import utilities as utils
import torch
from .other_methods import OtherResult


class StochasticApproximation(OtherResult):

    def __init__(self, network, c_vector, domain, primal_norm='linf', device='cpu', use_c_vector=False):
        super(StochasticApproximation, self).__init__(network, c_vector, domain, primal_norm)
        self.value = torch.tensor([1e-18]).to(device)
        self.iteration_count = 0
        self.DEVICE = torch.device(device)
        self.network = self.network.to(self.DEVICE)
        self.eval_list = []
        self.use_c_vector = use_c_vector

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

    def compute(self, max_iter=10000, track_evaluations=False, v=False, exact=None, tol=1e-5, mode="Absolute"):
        timer = utils.Timer()
        self.iteration_count = 0
        random_pts = self.domain.random_point(num_points=max_iter, requires_grad=True)

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
