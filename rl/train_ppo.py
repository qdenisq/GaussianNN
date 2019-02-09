import json
from pprint import pprint
import gym
from ppo import PPO
from models import SimplePPOAgent
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt


def train(*args, **kwargs):
    print(kwargs)

    device = 'cpu'
    kwargs['ppo']['device'] = device

    env = gym.make('Pendulum-v0')
    obs = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]

    # state_dim = env.get_state_dim()
    # action_dim = env.get_action_dim()
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = SimplePPOAgent(**kwargs['agent']).to(device)
    alg = PPO(agent=agent, **kwargs['ppo'])
    scores = alg.train(env, 200)

    agent.eval()
    for i in range(1):
        done = False
        state = env.reset(train_mode=True)
        rewards = []
        while not np.any(done):
            action, _, _, _ = agent(torch.from_numpy(state).float().to(device))
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            env.step(action)
            state, reward, done = env.step(action)
            rewards.append(reward)
            score = np.mean(np.asarray(rewards).sum(axis=0))
            print("\r play #{} | score: {}".format(i + 1, score), end='')

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = "../models/ppo_reacher_{}.pt".format(dt)
    torch.save(agent, model_fname)

    scores_fname = "../reports/ppo_reacher_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100)[:200])
    fig_name = "../reports/ppo_reacher_{}.png".format(dt)
    plt.savefig(fig_name)


if __name__ == '__main__':
    with open('config.json') as data_file:
        kwargs = json.load(data_file)
    pprint(kwargs)
    train(**kwargs)