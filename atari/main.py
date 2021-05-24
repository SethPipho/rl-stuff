import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym
import time
import matplotlib.pyplot as plt


env_name = 'Pong-v0'
n_episodes = 10000
episodes_per_update= 1
gamma = .95
lr = .001

log_interval = 1
demo_interval = 1000
save_interval = 5000

def preprocess_frame(frame, output_size=(84,84)) -> torch.Tensor:
    frame = torch.Tensor(frame) / 255.0
    frame = frame.permute(2, 0, 1)
    frame = torch.mean(frame, 0, keepdim=True)
    frame = torch.unsqueeze(frame, dim=0)
    frame = F.interpolate(frame, size=output_size, mode='bicubic', align_corners=True)
    frame = frame[0]
    return frame


class AtariActorCritic(nn.Module):
    def __init__(self):
        super(AtariActorCritic, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 256),
            nn.ReLU()
        )

        self.actor_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.features(x)
        return self.actor_head(feat), self.value_head(feat)



def run_episode(env, model, render=False, stop_on_done=True):
    #buffers
    probabilities = []
    actions = []
    rewards = []
    values = []

    obs = env.reset()
    prev_frame = preprocess_frame(obs) 
    cur_frame  = preprocess_frame(obs)
    while True:
        if render:
            env.render()
        
        model_input = torch.cat([prev_frame, cur_frame], dim=0)
        model_input = torch.unsqueeze(model_input, dim=0)
        prob, value = model(model_input)
        dist = Categorical(prob)
        action = dist.sample().numpy()[0]

        obs, reward, done, info = env.step(action)

        prev_frame = cur_frame
        cur_frame = preprocess_frame(obs)

        probabilities.append(prob)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        if done:
            break

    probabilities = torch.cat(probabilities)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    values = torch.stack(values)
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


if __name__ == "__main__":


    env = gym.make(env_name)
    model = AtariActorCritic()

    print(env.action_space)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    returns = []
    for i in range(0, n_episodes):
    
        optimizer.zero_grad()

        probabilities, actions, rewards, values = run_episode(env, model)
        discounted_rewards = compute_sum_discounted_rewards(rewards, gamma)
        
        advantage = discounted_rewards - values
        advantage =  (advantage - advantage.mean()) / advantage.std().clamp_min(1e-12)

        dist = Categorical(probabilities)
        log_prob = dist.log_prob(actions)
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = (advantage ** 2).mean()
        
        loss = actor_loss + critic_loss
        loss.backward()
        optimizer.step()

        returns.append(rewards.sum())

        if i % log_interval == 0:
            return_moving_avg = sum(returns[-log_interval:]) / min(log_interval, len(returns))
            print(f"n_episodes: {i} avg_reward: {return_moving_avg} best: {max(returns)}")
            plt.plot(returns, marker='o', markersize=1, linestyle="None")
            plt.savefig("rewards-history.png")

        if i % demo_interval == 0:
            run_episode(env, model, render=True)
            #need to close and remake env to close render window
            env.close()
            env = gym.make(env_name)

        if i % save_interval == 0 and i != 0:
            model_path = f"models/{env_name}-{str(i)}-model.pt"
            torch.save(policy_model, model_path)
            


    run_episode(env, model, render=True)

    model_path = f"models/{env_name}-{str(n_episodes)}-model.pt"
    torch.save(policy_model, model_path)

    plt.plot(returns, marker='o', markersize=1, linestyle="None")
    plt.savefig("rewards-history.png")

    env.close()