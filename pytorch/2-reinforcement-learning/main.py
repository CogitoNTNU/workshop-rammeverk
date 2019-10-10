import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb
from dinkode import PolicyNet

def train(polinet, env):
    episode_durations = []
    mean_durations = []

    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        
        plt.plot(durations_t.numpy())

        memes = durations_t.unfold(0, len(durations_t), 1).mean(1).view(-1)
        mean_durations.append(memes)
        
        plt.plot(mean_durations)

        plt.pause(0.001)

    num_episode = 5000
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99

    optimizer = torch.optim.RMSprop(polinet.parameters(), lr=learning_rate)

    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0


    for e in range(num_episode):

        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        env.render(mode='rgb_array')

        for t in count():
            probs = polinet(state)
            m = Bernoulli(probs)
            action = m.sample()

            action = action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array')

            if done:
                reward = 0

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        if e > 0 and e % batch_size == 0:

            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = polinet(state)
                m = Bernoulli(probs)
                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    polinet = PolicyNet()
    try:
        train(polinet, env)
    except:
        print('Your neural network is not working properly :(')
    