import json
from pprint import pprint
import torch
import numpy as np
import argparse
from src.environment import ReacherEnvironment


def play(*args, **kwargs):
    print(kwargs)

    fname = args[0]
    device = 'cpu'
    env = ReacherEnvironment(**kwargs['env'])

    agent = torch.load(fname).to(device)
    print(agent)
    agent.eval()
    for i in range(1):
        done = False
        state = env.reset(train_mode=False)
        rewards = []
        while not np.any(done):
            if 'ppo' in fname:
                action, _, _, _ = agent(torch.from_numpy(state).float().to(device))
            elif 'ddpg' in fname:
                action = agent.act(torch.from_numpy(state).float().to(device))
            else:
                raise ValueError('Unknown agent type. Please make sure that fname has either "ppo" or "ddpg".')
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            state, reward, done = env.step(action)
            rewards.append(reward)
            score = np.mean(np.asarray(rewards).sum(axis=0))
            print("\r play #{} | score: {}".format(i+1, score), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_fname', type=str, help='file containing the trained agent')
    args = parser.parse_args()
    agent_fname = args.agent_fname

    with open('../config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data
    play(agent_fname, **kwargs)