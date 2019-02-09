import numpy as np
import torch


class PPO:
    def __init__(self, agent=None, **kwargs):
        self.agent = agent

        self.actor_optim = torch.optim.Adam(agent.get_actor_parameters(), lr=kwargs['actor_lr'], eps=kwargs['learning_rate_eps'])
        self.critic_optim = torch.optim.Adam(agent.get_critic_parameters(), lr=kwargs['critic_lr'], eps=kwargs['learning_rate_eps'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']
        self.discount = kwargs['discount']
        self.lmbda = kwargs['lambda']
        self.minibatch_size = kwargs['minibatch_size']
        self.epsilon = kwargs['epsilon']
        self.beta = kwargs['beta']
        self.clip_grad = kwargs['clip_grad']
        self.device = kwargs['device']

    def rollout(self, env):
        """
           Runs an agent in the environment and collects trajectory
           :param env: Environment to run the agent in (ReacherEnvironment)
           :return states: (torch.Tensor)
           :return actions: (torch.Tensor)
           :return rewards: (torch.Tensor)
           :return dones: (torch.Tensor)
           :return values: (torch.Tensor)
           :return old_log_probs: (torch.Tensor)
           """
        state = env.reset()
        # Experiences
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []

        self.agent.eval()
        # Rollout
        while True:
            action, old_log_prob, _, value = self.agent(torch.from_numpy(state).float().to(self.device))
            action = np.clip(action.detach().cpu().numpy(), -1., 1.)
            _, old_log_prob, _, _ = self.agent(torch.from_numpy(state).float().to(self.device), torch.from_numpy(action).float().to(self.device))

            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.detach().cpu().numpy())
            old_log_probs.append(old_log_prob.detach().cpu().numpy())

            state = next_state

            if np.any(done):
                break

        states = torch.from_numpy(np.asarray(states)).float().to(self.device)
        actions = torch.from_numpy(np.asarray(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.asarray(dones).astype(int)).long().to(self.device)
        values = torch.from_numpy(np.asarray(values)).float().to(self.device)
        values = torch.cat([values, torch.zeros(1, values.shape[1], 1).to(self.device)], dim=0)
        old_log_probs = torch.from_numpy(np.asarray(old_log_probs)).float().to(self.device)

        return states, actions, rewards, dones, values, old_log_probs

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        scores = []
        for episode in range(num_episodes):
            states, actions, rewards, dones, values, old_log_probs = self.rollout(env)

            score = rewards.sum(dim=0).mean()

            T = rewards.shape[0]
            last_advantage = torch.zeros((rewards.shape[1], 1))
            last_return = torch.zeros(rewards.shape[1])
            returns = torch.zeros(rewards.shape)
            advantages = torch.zeros(rewards.shape)

            # calculate return and advantage
            for t in reversed(range(T)):
                # calc return
                last_return = rewards[t] + last_return * self.discount * (1 - dones[t]).float()
                returns[t] = last_return

            # Update
            returns = returns.view(-1, 1)
            states = states.view(-1, env.get_state_dim())
            actions = actions.view(-1, env.get_action_dim())
            old_log_probs = old_log_probs.view(-1, 1)

            # update critic
            num_updates = actions.shape[0] // self.minibatch_size
            self.agent.train()
            for k in range(self.num_epochs_critic):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    returns_batch = returns[idx]
                    states_batch = states[idx]

                    _, _, _, values_pred = self.agent(states_batch)

                    critic_loss = torch.nn.MSELoss()(values_pred, returns_batch)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.get_critic_parameters(), self.clip_grad)
                    self.critic_optim.step()

            # calc advantages
            self.agent.eval()
            for t in reversed(range(T)):
                # advantage
                next_val = self.discount * values[t + 1] * (1 - dones[t]).float()[:, np.newaxis]
                delta = rewards[t][:, np.newaxis] + next_val - values[t]
                last_advantage = delta + self.discount * self.lmbda * last_advantage
                advantages[t] = last_advantage.squeeze()

            advantages = advantages.view(-1, 1)
            advantages = (advantages - advantages.mean()) / advantages.std()

            # update actor
            self.agent.train()
            for k in range(self.num_epochs_actor):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    advantages_batch = advantages[idx]
                    old_log_probs_batch = old_log_probs[idx]
                    states_batch = states[idx]
                    actions_batch = actions[idx]

                    _, new_log_probs, entropy, _ = self.agent(states_batch, actions_batch)

                    ratio = (new_log_probs.view(-1, 1) - old_log_probs_batch).exp()
                    obj = ratio * advantages_batch
                    obj_clipped = ratio.clamp(1.0 - self.epsilon,
                                              1.0 + self.epsilon) * advantages_batch
                    entropy_loss = entropy.mean()

                    policy_loss = -torch.min(obj, obj_clipped).mean(0) - self.beta * entropy_loss

                    self.actor_optim.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.get_actor_parameters(), self.clip_grad)
                    self.actor_optim.step()

            scores.append(score)
            print("episode: {} | score:{:.4f} | action_mean: {:.2f}, action_std: {:.2f}".format(
                episode, score, actions.mean().cpu(), actions.std().cpu()))

        print("Training finished. Result score: ", score)
        return scores
