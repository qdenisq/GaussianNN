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
        sns.jointplot("x0", "x1", data, kind='kde');


def propagate(dists, x0, num_outputs):
    x1_idx = torch.linspace(0, 1, num_outputs)
    x0_idx = torch.linspace(0, 1, x0.shape[0])

    x1 = x1_idx.repeat(5)

    x0v, x1v = torch.meshgrid([x0_idx, x1_idx])
    points = torch.stack([x0v, x1v], dim=2)
    points = points.view(-1, 2)
    # log probs
    log_probs = dists.log_prob(points)





    return None



if __name__ == '__main__':
    num_input_units = 10
    num_output_units = 5
    x0 = random_activation(num_input_units)

    ticks = np.linspace(0, 1, num_input_units)
    plt.figure()
    sns.lineplot(ticks, x0)

    d, mu, sigma = generate_GMM(3)
    plt.figure()
    plot_gmm(d)

    propagate(d, x0, num_output_units)


    plt.show()




