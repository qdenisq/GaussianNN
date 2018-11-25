import torch
from torch.nn import Module, Parameter
from torch.distributions import Normal
import math


class GaussianLayer(Module):
    def __init__(self, in_features, out_features, num_components, sigma_gamma):
        super(GaussianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.sigma_gamma = sigma_gamma

        self.mus = Parameter(torch.rand(num_components, 2))
        # self.log_vars = Parameter(-5. - torch.rand(num_components, 2) / (2 * torch.sqrt(torch.Tensor([float(num_components)]))) * sigma_gamma)
        self.log_vars = Parameter(-3. - torch.rand(num_components, 2))
        self.weights = Parameter((torch.rand(num_components)) / (num_components))

        self.x_in_idx = torch.linspace(0, 1, self.in_features)
        self.x_out_idx = torch.linspace(0, 1, self.out_features)


    def forward(self, x):
        vars = self.log_vars.exp()
        sigmas = vars.sqrt()

        x_out = torch.zeros(x.shape[0], self.out_features)

        # compute Cs
        C = torch.zeros(x.shape[0], self.num_components)
        for m in range(self.num_components):
            mu_x0 = self.mus[m, 0]
            deltas = (self.x_in_idx - mu_x0).pow(2)
            exp_terms = -deltas / (2 * vars[m, 0])
            # smth = torch.mm(x, exp_terms.exp().reshape(-1,1))
            C[:, m] = torch.mm(x, exp_terms.exp().reshape(-1,1)).squeeze() / torch.sqrt(2. * math.pi * vars[m, 0]) * self.weights[m]

        for j in range(self.out_features):
            # calc prob of x1_i along x1 axis by summing up probs of x1_i given each of m Gaussian components
            dist = Normal(self.mus[:, 1], sigmas[:, 1])  # X1 marginal
            log_probs = dist.log_prob(self.x_out_idx[j])
            probs = log_probs.exp()
            weighted_probs = probs * C

            result_prob = weighted_probs.sum(dim=-1)
            x_out[:, j] = result_prob / (self.in_features * self.weights.sum())

        return x_out

