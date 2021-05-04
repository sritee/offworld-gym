#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

import gym
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(env_name='Pendulum-v0', seed=None):
    # hacky
    # seeds need to be passed into dm control wrapper creation
    if type(env_name) == str:
        return gym.make(env_name)
    else:
        return dmc2gym.make(env_name[0], task_name=env_name[1], seed=seed)
    
    
def eval_policy(policy, eval_env, seed, eval_episodes=10, render=False):
    
    returns = []
    for idx in range(eval_episodes):
        cum_reward = 0
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            if reward == -1: # hack during eval
                reward = 0
            cum_reward += reward
            if render and idx == eval_episodes - 1:
                eval_env.render()
        returns.append(cum_reward)

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    print("---------------------------------------")
    print(f"return over {eval_episodes} episodes:" 
       f"mean {mean_return:.3f}, std {std_return:.3f} , min {min_return}, max {max_return}")
    print("---------------------------------------")
    return mean_return, std_return, min_return, max_return

def collect_policy_buffer(policy, env_name, seed, num_timesteps = 1e5, render=False):
    
    env = make_env(env_name, seed=seed)
    env.seed(seed + 100)
    buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0])
    returns = []
    t = 0
    num_episodes =0
    while t < num_timesteps:
        cum_reward = 0
        state, done = env.reset(), False
        ep_steps = 0
        num_episodes += 1
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, info = env.step(action)
            terminal = done and 'TimeLimit.truncated' not in info # time clipped episodes are non-terminal
            buffer.add(state, action, next_state, reward, 1 - terminal)
            state = next_state
            cum_reward += reward
            if render:
                env.render()
            ep_steps += 1
        t+= ep_steps
        if num_episodes % 10 == 0:
            print(f'collected {num_episodes} episodes, {t} timesteps')
        returns.append(cum_reward)
    
    print(f'collected buffer of size {buffer.size}, avg reward {np.mean(returns)}')
    
    return buffer
    
def debug_critic(policy, env_name, seed, gamma, eval_episodes = 1):
    
    states = []
    returns = []
    critic_values = []
    
    eval_env = make_env(env_name, seed=seed)
    eval_env.seed(seed + 100)

    for idx in range(eval_episodes):
        cum_reward = []
        state, done = eval_env.reset(), False
        start_state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        start_action = torch.FloatTensor(policy.select_action(np.array(state))).reshape(1, -1).to(DEVICE)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            cum_reward.append(reward)
        
        discounted_return = 0
        # [1, 2, 3, 4, 5]
        # [1, 0.99, ....0.99^4]
        for idx in range(len(cum_reward) - idx - 1, -1, -1): ## iterate backwards
            discounted_return = discounted_return * gamma + cum_reward[idx]
        returns.append(discounted_return)
        
        critic_values.append(policy.critic.Q1(start_state, start_action).item())
    

    
    print(f'critic start pred {critic_values[-1]}, actual {returns[-1]}')
    
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_dim), dtype='float32')
        self.action = np.zeros((max_size, action_dim), dtype='float32')
        self.next_state = np.zeros((max_size, *state_dim), dtype='float32')
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, device=DEVICE):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
def save_buffer(replay_buffer, name):
    with open(name, 'wb') as f:
        pickle.dump(replay_buffer, f)

def load_buffer(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
    