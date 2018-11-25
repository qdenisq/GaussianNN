import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def random_activation(n_units):
    act = torch.rand(n_units)
    return act


def generate_GMM(num_components):
    mus = torch.rand(num_components, 2)
    sigmas = torch.rand(num_components, 2).sqrt()/10
    dists = torch.distributions.Normal(mus, sigmas)
    return dists, mus, sigmas


def plot_gmm(dists):
    mus = dists.mean
    sigmas = dists.stddev

    data = dists.sample_n(2000).detach().cpu().numpy().reshape(-1,2)

    # data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    data = pd.DataFrame(data, columns=['x0', 'x1'])
    with sns.axes_style('white'):
        sns.jointplot("x0", "x1", data, kind='kde', title='weights');


def propagate(dists, x0, num_outputs):
    x1 = torch.zeros(num_outputs)

    x1_idx = torch.linspace(0, 1, num_outputs)
    x0_idx = torch.linspace(0, 1, x0.shape[0])



    x0v, x1v = torch.meshgrid([x0_idx, x1_idx])
    points = torch.stack([x0v, x1v], dim=2)
    # points = points.view(-1, 2)
    for j in range(x1_idx.shape[0]):
        for i in range(x0_idx.shape[0]):
            point = points[i, j]
            # log probs
            log_probs = dists.log_prob(point)
            probs = log_probs.sum(dim=-1).exp()

            weighted_probs = probs * x0[i]

            x1[j] += weighted_probs.sum()
    return x1



if __name__ == '__main__':
    num_input_units = 100
    num_output_units = 100
    x0 = random_activation(num_input_units)
    # x0 = torch.ones(num_input_units)

    palette = sns.color_palette()
    plt.figure('input')
    sns.barplot(np.round(np.linspace(0, 1, num_input_units), 2), x0, color=palette[0])

    d, mu, sigma = generate_GMM(20)
    plot_gmm(d)

    x1 = propagate(d, x0, num_output_units)

    plt.figure('output')
    sns.barplot(np.round(np.linspace(0, 1, num_output_units), 2), x1, color=palette[1])

    x1_act = torch.nn.Tanh()(x1)

    plt.figure('output after activation')
    sns.barplot(np.round(np.linspace(0, 1, num_output_units), 2), x1_act, color=palette[1])

    plt.show()




