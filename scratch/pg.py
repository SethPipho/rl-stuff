import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import time
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'

n_episodes = 30000
episodes_per_update= 1
gamma = .99
lr = .002

env = gym.make(env_name)
obs_n = env.observation_space.shape[0]
action_n = env.action_space.n

print(obs_n, action_n)

model = nn.Sequential(
            nn.Linear(obs_n,5), 
            nn.ReLU(),
            nn.Linear(5, action_n),
            nn.Softmax(dim=1)
        )


def run_episode(render=False, stop_on_done=True):
    #buffers
    observations = []
    actions = []
    rewards = []

    obs = env.reset()
    while True:
        if render:
            env.render()
            
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float) 
            observations.append(obs_t)
            obs_t = torch.unsqueeze(obs_t, dim=0)
            logits = model(obs_t)
            dist = Categorical(logits)
            action = dist.sample().numpy()[0]

        obs, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

        if done:
            break

    observations = torch.stack(observations)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    return observations, actions, rewards


def compute_discounted_rewards(rewards, gamma):
    ''' computes the cumulative discounted rewards '''
    n = len(rewards)
    disc_rewards = [0 for i in range(n)]
    cum = 0
    for i in range(n-1, -1, -1):
        cum = rewards[i] + (cum * gamma)
        disc_rewards[i] = cum 
    
    disc_rewards = torch.tensor(disc_rewards)
    return disc_rewards


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
reward_history = []

for i in range(0, n_episodes, episodes_per_update):
    optimizer.zero_grad()
    observations = [] 
    actions = []
    rewards = []

    for j in range(episodes_per_update):
        ep_observations, ep_actions, ep_rewards = run_episode()
       
        ep_discounted_reqards = compute_discounted_rewards(ep_rewards, gamma)

        observations.append(ep_observations)
        actions.append(ep_actions)
        rewards.append(ep_discounted_reqards)

        reward_history.append(ep_rewards.sum().numpy())

    observations = torch.cat(observations)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)

    rewards =  (rewards - rewards.mean()) / rewards.std().clamp_min(1e-12)

    logits = model(observations)
    dist = Categorical(logits)
    log_prob = dist.log_prob(actions)
    loss = -(log_prob * rewards).mean()
    
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        reward_moving_avg = sum(reward_history[-10:]) / min(10, len(reward_history))
        print(f"n_episodes: {i} avg_reward: {reward_moving_avg}")
    
    if i % 500 == 0:
        env = gym.make(env_name)
        run_episode(render=True)
        env.close()
        env = gym.make(env_name)
        plt.plot(reward_history, marker='o', markersize=1, linestyle="None")
        plt.savefig("rewards.png")

    if i % 5000 == 0:
        torch.save(model, env_name + "-" + str(i) + "-model.pt")


    #print("loss:", loss.detach().numpy())


run_episode(render=True, stop_on_done=False)

torch.save(model, env_name + "-" + str(n_episodes) + "-model.pt")

plt.plot(reward_history, marker='o', markersize=1, linestyle="None")
plt.savefig("rewards.png")

env.close()