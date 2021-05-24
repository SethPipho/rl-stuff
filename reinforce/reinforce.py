import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import time
import matplotlib.pyplot as plt


env_name = 'LunarLander-v2'
n_episodes = 10000
gamma = .99
lr = .002

log_interval = 100
demo_interval = 1000
save_interval = 5000


def run_episode(env, model, render=False, stop_on_done=True):
    #buffers
    probabilities = []
    actions = []
    rewards = []

    obs = env.reset()
    while True:
        if render:
            env.render()
        obs = torch.tensor(obs, dtype=torch.float)  
        obs = torch.unsqueeze(obs, dim=0)
        prob = model(obs)
        dist = Categorical(prob)
        action = dist.sample().numpy()[0]

        obs, reward, done, info = env.step(action)

        probabilities.append(prob)
        actions.append(action)
        rewards.append(reward)

        if done:
            break

    probabilities = torch.cat(probabilities)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    return probabilities, actions, rewards


def compute_sum_discounted_rewards(rewards, gamma):
    ''' computes the cumulative discounted rewards '''
    n = len(rewards)
    disc_rewards = [0 for i in range(n)]
    cum = 0
    for i in range(n-1, -1, -1):
        cum = rewards[i] + (cum * gamma)
        disc_rewards[i] = cum 
    
    disc_rewards = torch.tensor(disc_rewards)
    return disc_rewards


#main loop

env = gym.make(env_name)
obs_n = env.observation_space.shape[0]
action_n = env.action_space.n

print(obs_n)


model = nn.Sequential(
            nn.Linear(obs_n,5),
            nn.ReLU(), 
            nn.Linear(5, action_n),
            nn.Softmax(dim=1)
        )

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
reward_history = []

for i in range(0, n_episodes):
   
    optimizer.zero_grad()
    probabilities, actions, rewards = run_episode(env, model)
    discounted_rewards = compute_sum_discounted_rewards(rewards, gamma)
    discounted_rewards =  (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std().clamp_min(1e-12)
    dist = Categorical(probabilities)
    log_prob = dist.log_prob(actions)
    loss = -(log_prob * discounted_rewards).mean()
    loss.backward()
    optimizer.step()

    reward_history.append(rewards.sum())

    if i % log_interval == 0:
        reward_moving_avg = sum(reward_history[-log_interval:]) / min(log_interval, len(reward_history))
        print(f"n_episodes: {i} avg_reward: {reward_moving_avg}")
    
    if i % demo_interval == 0:
        run_episode(env, model, render=True)
        plt.plot(reward_history, marker='o', markersize=1, linestyle="None")
        plt.savefig("rewards-history.png")

        #need to close and remake env to close render window
        env.close()
        env = gym.make(env_name)

    if i % save_interval == 0 and i != 0:
        model_path = f"{env_name}-{str(i)}-model.pt"
        torch.save(model, model_path)


run_episode(env, model, render=True)

model_path = f"{env_name}-{str(n_episodes)}-model.pt"
torch.save(policy_model, model_path)

plt.plot(reward_history, marker='o', markersize=1, linestyle="None")
plt.savefig("rewards-history.png")

env.close()