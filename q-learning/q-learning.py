import gym
import numpy as np
import random

env_name = "FrozenLake-v0"
env = gym.make(env_name)

obs_n = env.observation_space.n 
action_n = env.action_space.n
print(obs_n, action_n)

q_table = np.zeros((obs_n, action_n))

epsilon = 1.0
decay = .01
gamma = 1.0
lr = 0.1
n_episodes = 1000
log_every = 100

def moving_average(arr, n):
    return sum(arr[-n:])/min(n, len(arr))

reward_history = []
for i in range(1000):
    reward_sum = 0
    obs = env.reset()
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs][:])
        
        next_obs, reward, done, info = env.step(action)
        
        reward_sum += reward
        td = (reward + gamma * np.max(q_table[next_obs])) - q_table[obs][action]
        q_table[obs][action] += lr * td
        
        obs = next_obs
        if done:
            break

    reward_history.append(reward_sum)
    if i % log_every == 0:
        print(f"{i} reward: {moving_average(reward_history, log_every)}")
        print(q_table)
    epsilon *= (1 - decay) 

        