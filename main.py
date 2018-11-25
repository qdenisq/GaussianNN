import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import math
import time

def random_activation(n_units):
    act = torch.rand(n_units)
    return act


def generate_GMM(num_components, sigma_gamma=1.):
    mus = torch.rand(num_components, 2)
    sigmas = torch.rand(num_components, 2) / (2. * torch.sqrt(torch.tensor([float(num_components)]))) * sigma_gamma
    weights = torch.rand(num_components)
    dists = torch.distributions.Normal(mus, sigmas)
    return dists, mus, sigmas, weights


def plot_gmm(dists, weights):
    mus = dists.mean
    sigmas = dists.stddev
    data = []
    for i in range(mus.shape[0]):
        dist = torch.distributions.Normal(mus[i], sigmas[i])
        data.append(dist.sample((int(weights[i].item()*1000),)))
    data = torch.cat(data).detach().cpu().numpy()
    # data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x0', 'x1'])
    with sns.axes_style('white'):
        sns.jointplot("x0", "x1", data, kind='kde')
        plt.title('weights')

def propagate(mus, sigmas, weights,  x0, num_outputs):
    x1 = torch.zeros(num_outputs)
    vars = sigmas.pow(2)


    x1_idx = torch.linspace(0, 1, num_outputs)
    x0_idx = torch.linspace(0, 1, x0.shape[0])

    # compute Cs
    num_inputs = x0.shape[0]
    num_components = weights.shape[0]
    C = torch.zeros(num_components)
    for m in range(num_components):
        mu_x0 = mus[m, 0]
        deltas = (x0_idx - mu_x0).pow(2)
        exp_terms = -deltas / (2*vars[m,0])
        C[m] = (exp_terms.exp() * x0).sum() / torch.sqrt(2. * math.pi * vars[m,0]) * weights[m]

    for j in range(x1_idx.shape[0]):
        # calc prob of x1_i along x1 axis by summing up probs of x1_i given each of m Gaussian components
        dist = torch.distributions.Normal(mus[:,1], sigmas[:,1]) # X1 marginal
        log_probs = dist.log_prob(x1_idx[j])
        probs = log_probs.exp()
        weighted_probs = probs * C

        result_prob = weighted_probs.sum()
        x1[j] += result_prob/(num_inputs * weights.sum())


    return x1



if __name__ == '__main__':
    num_input_units = 100
    num_output_units = 100
    x0 = random_activation(num_input_units)
    # x0 = x0 * torch.linspace(0, 1, num_input_units)
    # x0 = torch.ones(num_input_units)

    palette = sns.color_palette()
    plt.figure('input')
    sns.barplot(np.round(np.linspace(0, 1, num_input_units), 2), x0, color=palette[0])

    d, mus, sigmas, weights = generate_GMM(100, 0.1)
    plot_gmm(d, weights)

    t0 = time.time()
    x1 = propagate(mus, sigmas, weights, x0, num_output_units)
    t1 = time.time()
    print(t1-t0)


    plt.figure('output')
    sns.barplot(np.round(np.linspace(0, 1, num_output_units), 2), x1, color=palette[1])

    x1_act = torch.nn.Tanh()(x1)

    plt.figure('output after activation')
    sns.barplot(np.round(np.linspace(0, 1, num_output_units), 2), x1_act, color=palette[1])

    plt.show()




