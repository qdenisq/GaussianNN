import numpy as np
from numpy.random import normal, uniform
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import math

def sample_from_wgmm(weights, mus, sigmas):
     # find all components with positive weights

    pos_idx = [j for j, e in enumerate(weights) if e >= 0]
    neg_idx = [j for j, e in enumerate(weights) if e < 0]
    cumsum = np.cumsum([weights[j] for j in pos_idx])
    accepted = False
    while not accepted:
        # choose a component to generate from
        rv = uniform(low=0.0, high=cumsum[-1].detach().cpu().numpy(), size=1)
        idx = np.searchsorted(cumsum, rv)
        idx = pos_idx[idx[0]]
        # generate rnd number
        x = sample(mus[idx,:], sigmas[idx,:]) # pass mu and sigma of the component to sample from

        # find probability for given rnd number
        mus_x0 = mus[:,0]
        deltas_x0 = (x[0] - mus_x0).pow(2)
        log_prob_x0 = -deltas_x0 / (2 * sigmas.pow(2)[:, 0])

        mus_x1 = mus[:, 1]
        deltas_x1 = (x[1] - mus_x1).pow(2)
        log_prob_x1 = -deltas_x1 / (2 * sigmas.pow(2)[:, 1])

        log_prob_x1 = log_prob_x1.unsqueeze_(1)

        log_probs = log_prob_x1 + log_prob_x0

        probs = (log_probs.exp() * weights)

        pos_probs = torch.clamp(probs, min=0).sum()
        neg_probs = torch.clamp(probs, max=0).sum()

        # x1 = torch.mm(x, probs.transpose(1, 0))



        # p_pos = sum([self.gmm_pdf[j](cf) for j in pos_idx])
        # p_neg = sum([self.gmm_pdf[j](cf) for j in neg_idx])
        # p = sum([self.gmm_pdf[j](cf) for j in range(len(self.gmm_pdf))])
        if uniform(0.0, pos_probs.detach()) > -1. * neg_probs.detach():
            accepted = True
        return x


def sample(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    return dist.sample()


def plot_gmm(weights, mus, sigmas, num_samples=2000):
    data = [sample_from_wgmm(weights, mus, sigmas) for _ in range(num_samples)]
    data = torch.stack(data, dim=0).detach().cpu().numpy()
    # data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x0', 'x1'])
    with sns.axes_style('white'):
        sns.jointplot("x0", "x1", data, kind='kde')
        plt.title('weights')