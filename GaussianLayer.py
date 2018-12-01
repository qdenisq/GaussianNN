import torch
from torch.nn import Module, Parameter
from torch.distributions import Normal
import math
import time

torch.manual_seed(0)

class GaussianLayer(Module):
    def __init__(self, in_features, out_features, num_components, sigma_gamma):
        super(GaussianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.sigma_gamma = sigma_gamma

        self.mus = Parameter(torch.rand(num_components, 2))
        # self.log_vars = Parameter(-5. - torch.rand(num_components, 2) / (2 * torch.sqrt(torch.Tensor([float(num_components)]))) * sigma_gamma)
        log_var = (1 / torch.sqrt(torch.Tensor([float(num_components)]))).pow(2).log()

        self.log_vars = Parameter(log_var - torch.rand(num_components, 2))
        self.weights = Parameter(((torch.rand(num_components))-0.5))
        self.bias = Parameter((torch.rand(out_features)) - 0.5)

        self.x_in_idx = torch.linspace(0, 1, self.in_features)
        self.x_out_idx = torch.linspace(0, 1, self.out_features)


    def forward(self, x):
        self.x_in_idx = self.x_in_idx.to(self.device)
        self.x_out_idx = self.x_out_idx.to(self.device)

        vars = self.log_vars.exp()
        sigmas = vars.sqrt()

        x_out = torch.zeros(x.shape[0], self.out_features)

        # compute Cs
        t0 = time.time()

        C = torch.zeros(x.shape[0], self.num_components)

        for m in range(self.num_components):
            mu_x0 = self.mus[m, 0]
            deltas = (self.x_in_idx - mu_x0).pow(2)
            exp_terms = -deltas / (2 * vars[m, 0])
            # smth = torch.mm(x, exp_terms.exp().reshape(-1,1))
            C[:, m] = torch.mm(x, exp_terms.exp().reshape(-1,1)).squeeze() / torch.sqrt(2. * math.pi * vars[m, 0]) * self.weights[m]

        t1 = time.time()

        for j in range(self.out_features):
            # calc prob of x1_i along x1 axis by summing up probs of x1_i given each of m Gaussian components
            dist = Normal(self.mus[:, 1], sigmas[:, 1])  # X1 marginal
            log_probs = dist.log_prob(self.x_out_idx[j])
            probs = log_probs.exp()
            weighted_probs = probs * C

            result_prob = weighted_probs.sum(dim=-1)
            x_out[:, j] = result_prob / (self.in_features * self.weights.sum())

        t2 = time.time()
        print(x.shape[0], t2-t0, t1-t0, t2 - t1)

        return x_out


class GaussianLayer1(Module):
    def __init__(self, in_features, out_features, num_components, sigma_gamma):
        super(GaussianLayer1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.sigma_gamma = sigma_gamma

        self.mus = Parameter(torch.rand(num_components, 2))
        # self.log_vars = Parameter(-5. - torch.rand(num_components, 2) / (2 * torch.sqrt(torch.Tensor([float(num_components)]))) * sigma_gamma)
        log_var = (1 / torch.sqrt(torch.Tensor([float(num_components)]))).pow(2).log()
        self.log_vars = Parameter(log_var - torch.rand(num_components, 2))
        self.weights = Parameter((torch.rand(num_components)-0.5))
        self.bias = Parameter((torch.rand(out_features))-0.5)
        self.x_in_idx = torch.linspace(0, 1, self.in_features)
        self.x_out_idx = torch.linspace(0, 1, self.out_features)

    def forward(self, x):
        self.x_in_idx = self.x_in_idx.to(self.mus.device)
        self.x_out_idx = self.x_out_idx.to(self.mus.device)

        vars = self.log_vars.exp()
        sigmas = vars.sqrt()

        t0 = time.time()

        mus_x0 = self.mus[:,0]
        deltas_x0 = (self.x_in_idx.view(-1, 1) - mus_x0).pow(2)
        log_prob_x0 = -deltas_x0 / (2 * vars[:, 0])

        mus_x1 = self.mus[:, 1]
        deltas_x1 = (self.x_out_idx.view(-1, 1) - mus_x1).pow(2)
        log_prob_x1 = -deltas_x1 / (2 * vars[:, 1])

        log_prob_x1 = log_prob_x1.unsqueeze_(1)

        log_probs = log_prob_x1 + log_prob_x0

        probs = (log_probs.exp() * self.weights).sum(dim=-1)

        x1 = torch.mm(x, probs.transpose(1, 0)) + self.bias

        t1 = time.time()
        # print(t1-t0)
        return x1


class GaussianLayer2(Module):
    def __init__(self, in_features, out_features, num_components, sigma_gamma):
        super(GaussianLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components
        self.sigma_gamma = sigma_gamma

        self.mus = Parameter(torch.rand(num_components, 2))
        # self.log_vars = Parameter(-5. - torch.rand(num_components, 2) / (2 * torch.sqrt(torch.Tensor([float(num_components)]))) * sigma_gamma)
        log_var = (1 / torch.sqrt(torch.Tensor([float(num_components)]))).pow(2).log()
        self.log_vars = Parameter(log_var - torch.rand(num_components, 2))
        self.weights = Parameter(((torch.rand(num_components))-0.5))

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