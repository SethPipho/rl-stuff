import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import time
import matplotlib.pyplot as plt


env_name = 'LunarLander-v2'
n_episodes = 10000
episodes_per_update= 1
gamma = .99
lr = .001

log_interval = 100
demo_interval = 1000
save_interval = 1000


def run_episode(env, policy_model, value_model, render=False, stop_on_done=True):
    #buffers
    probabilities = []
    actions = []
    rewards = []
    values = []

    obs = env.reset()
    while True:
        if render:
            env.render()
        obs = torch.tensor(obs, dtype=torch.float) 
        obs = torch.unsqueeze(obs, dim=0)
        prob = policy_model(obs)
        dist = Categorical(prob)
        action = dist.sample().numpy()[0]
        value = value_model(obs)[0]

        obs, reward, done, info = env.step(action)

        probabilities.append(prob)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        if done:
            break

    probabilities = torch.cat(probabilities)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    values = torch.cat(values)
    return probabilities, actions, rewards, values


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

policy_model = nn.Sequential(
            nn.Linear(obs_n,5), 
            nn.ReLU(),
            nn.Linear(5, action_n),
            nn.Softmax(dim=1)
        )

value_model = nn.Sequential(
            nn.Linear(obs_n,5), 
            nn.ReLU(),
            nn.Linear(5, 1),
        )

policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=lr)
returns = []

for i in range(0, n_episodes):
   
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    
    probabilities, actions, rewards, values = run_episode(env, policy_model, value_model)
    discounted_rewards = compute_sum_discounted_rewards(rewards, gamma)
    
    advantage = discounted_rewards - values
    advantage =  (advantage - advantage.mean()) / advantage.std().clamp_min(1e-12)
    

    dist = Categorical(probabilities)
    log_prob = dist.log_prob(actions)
    policy_loss = -(log_prob * advantage.detach()).mean()
    policy_loss.backward()
    policy_optimizer.step()

    value_loss = (advantage ** 2).mean()
    
    value_loss.backward()
    value_optimizer.step()

    #print(next(value_model.parameters()).grad)

    returns.append(rewards.sum())

    if i % log_interval == 0:
        return_moving_avg = sum(returns[-log_interval:]) / min(log_interval, len(returns))
        print(f"n_episodes: {i} avg_reward: {return_moving_avg} best: {max(returns)}")
        plt.plot(returns, marker='o', markersize=1, linestyle="None")
        plt.savefig("rewards-history.png")

    if i % demo_interval == 0:
        run_episode(env, policy_model, value_model, render=True)
        #need to close and remake env to close render window
        env.close()
        env = gym.make(env_name)

    if i % save_interval == 0 and i != 0:
        model_path = f"models/{env_name}-{str(i)}-model.pt"
        torch.save(policy_model, model_path)
        


run_episode(env, policy_model, value_model, render=True)

model_path = f"models/{env_name}-{str(n_episodes)}-model.pt"
torch.save(policy_model, model_path)

plt.plot(returns, marker='o', markersize=1, linestyle="None")
plt.savefig("rewards-history.png")

env.close()