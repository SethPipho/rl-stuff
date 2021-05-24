import multiprocessing as mp
import gym
import time
import pybulletgym
import numpy as np

def gym_mp_worker(pid, env_name, conn):
    #print(pid, env.reset())
    env = gym.make(env_name)
    env.reset()
    while True:
        msg = conn.recv()
        if msg == "step":
            obs, reward, done, info = env.step(0)
            conn.send((obs, reward, done, info))
        if done:
           env.reset()
  
class VecEnv:
    def __init__(self, env_name, n):
        self.env_name = env_name
        self.processes = []
        self.conns = []

        for i in range(n):
            parent_conn, child_conn = mp.Pipe()
           
            process = mp.Process(target=gym_mp_worker, args=(i, env_name, child_conn))
            process.start()

            self.processes.append(process)
            self.conns.append(parent_conn)


    def step(self):
        observations = []
        rewards = []
        dones = []
        infos = []
        for conn in self.conns:
            conn.send("step")
        observations = []
        rewards = []
        for conn in self.conns:
            obs, reward, done, info = conn.recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos

    def close(self):
        for p in self.processes:
            p.terminate()
        for p in self.processes:
            p.join()

n = 1
steps = 5000

if __name__ == "__main__":

    start = time.time()
    env = gym.make("Walker2DMuJoCoEnv-v0")
    print(env.action_space)
    #env.render()
    env.reset()
    for i in range(steps):
       
        obs, reward, done, info = env.step(np.random.random(17))
        if done:
            env.reset()
    elapsed = time.time() - start
    print("elapsed:", elapsed)
    print("throughput (steps/sec):", steps * n/elapsed)

    env.close()
