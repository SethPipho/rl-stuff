import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import time
import matplotlib.pyplot as plt


env_name = 'LunarLander-v2'
env = gym.make(env_name)

policy_model = torch.load("models/LunarLander-v2-10000-model.pt")



def run_episode(env, policy_model, render=False, stop_on_done=True):
    #buffers
    rewards = []

    obs = env.reset()
    while True:
        if render:
            env.render()
        obs = torch.tensor(obs, dtype=torch.float) 
        obs = torch.unsqueeze(obs, dim=0)
        prob = policy_model(obs)
        #action = torch.argmax(prob).numpy()

        dist = Categorical(prob)
        action = dist.sample().numpy()[0]
    
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    
    rewards = torch.tensor(rewards)
    return rewards


rewards = run_episode(env, policy_model, render=True)
print(rewards.sum())
print(len(rewards))
