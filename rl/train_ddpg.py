import json
from pprint import pprint
import numpy as np
from src.environment import ReacherEnvironment
from src.ddpg import DDPG
from src.models import SimpleDDPGAgent
import datetime
import matplotlib.pyplot as plt
import torch


def train(*args, **kwargs):
    print(kwargs)

    env = ReacherEnvironment(**kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    kwargs['ddpg']['device'] = 'cpu'

    agent = SimpleDDPGAgent(**kwargs['agent'])
    target_agent = SimpleDDPGAgent(**kwargs['agent'])
    alg = DDPG(agent=agent, target_agent=target_agent, **kwargs['ddpg'])
    scores = alg.train(env, 200)

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = "../models/ddpg_reacher_{}.pt".format(dt)
    torch.save(agent, model_fname)

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100) / 100))
    fig_name = "../reports/ddpg_reacher_{}.png".format(dt)
    plt.savefig(fig_name)
    scores_fname = "../reports/ddpg_reacher_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))


if __name__ == '__main__':
    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)