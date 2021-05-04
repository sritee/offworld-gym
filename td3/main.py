#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import torch
import argparse
import os

from model import TD3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
from utils import ReplayBuffer, eval_policy

import logging
import time
import gym

env_name = 'OffWorldDockerMonolithContinuousSim-v0'
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if env_name in env:
         print("Remove {} from registry".format(env))
         del gym.envs.registration.registry.env_specs[env]

import offworld_gym
from offworld_gym.envs.common.channels import Channels
from ou_noise import OUNoise

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

#import functools
from collections import deque

class DummyEnv:

    def step(self):

        return np.zeros([1, 60, 80]), 0, 0, None

    def reset(self):
        return np.zeros([1, 60, 80])


class DownsampleWrapper:

    def __init__(self, env, downsample_factor=4, history = 4):

        self.env = env
        env.action_space.high *= 0.5
        env.action_space.low *= 0.5
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.downsample_factor = downsample_factor
        self.out_shape=(env.observation_space.shape[-1], int(env.observation_space.shape[1]/downsample_factor),
                        int(env.observation_space.shape[2]/downsample_factor))

        # hardcoded
        self._max_episode_steps = 100
        self.history = history
        self.reset_buffer()
        self.t = 0

    def step(self, action):

        try:
            s, r, d, i = self.env.step(action)
            self.t += 1
        except:
            self.env = gym.make("OffWorldDockerMonolithContinuousSim-v0", channel_type=Channels.DEPTH_ONLY, random_init=False)
            return self.step(action)

        if d and r != 1 and self.t != self._max_episode_steps:
            r = -1
        elif d:
            print(f'done in {self.t}, {r}')

        self.buffer.append(self.downsample_state(s))
        return self.obs_from_buffer(), r, d, i

    def reset(self):

        self.reset_buffer()
        try:
            s = self.env.reset()
        except:
            self.env = gym.make("OffWorldDockerMonolithContinuousSim-v0", channel_type=Channels.DEPTH_ONLY, random_init=False)
            return self.reset()
        self.buffer.append(self.downsample_state(s))
        self.t = 0
        return self.obs_from_buffer()

    def obs_from_buffer(self):

         return np.array(self.buffer).reshape(-1, self.out_shape[1], self.out_shape[2])

    def downsample_state(self, state):

        return state[0, ::self.downsample_factor, ::self.downsample_factor, :].transpose(2, 0, 1)

    def seed(self, seed):

        self.env.seed(seed)

    def close(self):

        self.env.close()

    def reset_buffer(self):

        self.buffer = deque(maxlen=self.history)
        for _ in range(self.history):
            self.buffer.append(np.zeros(self.out_shape))

    @property
    def state_shape(self):

        return (self.history * self.out_shape[0], self.out_shape[1], self.out_shape[2])

# create the environment
env = DownsampleWrapper(gym.make("OffWorldDockerMonolithContinuousSim-v0", channel_type=Channels.DEPTH_ONLY, random_init=False))
env.reset()
writer = SummaryWriter()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)       # How often (timesteps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.5)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.995)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.025)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--save_buffer_and_exit", type=bool)
    args = parser.parse_args()

    #args.env = 'Pendulum-v0'
    #args.env = ['cheetah', 'run']
    #args.env = 'HalfCheetah-v2'
    #args.env = ['hopper', 'hop']
    #args.env = 'LunarLanderContinuous-v2'
    args.max_timesteps = 1e7
    args.start_timesteps = 5000
    args.eval_freq = 300
    args.tau = 0.005
    args.batch_size = 128
    args.replay_buffer_size = int(70000)
    args.save_model = True
    #args.load_model = 'TD3_halfcheetahv2_expert/TD3_HalfCheetah-v2_0'

    file_name = f"{args.policy}_{args.seed}"
    print("---------------------------------------")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    #env = make_env(args.env, seed=args.seed)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.state_shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)

    #if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load('./models/TD3_0')



    # Evaluate untrained policy
    evaluations = []
    evaluations.append(eval_policy(policy, env, args.seed, render=False, eval_episodes=50))

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=args.replay_buffer_size)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    actor_loss, critic_loss = None, None
    noise = OUNoise(2, sigma=0.35)

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if np.random.rand() > args.expl_noise:
            action = policy.select_action(np.array(state))
        else:
            action = np.clip(noise.sample(), env.action_space.low, env.action_space.high)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        args.expl_noise = ((0.1 - 0.9)/20000) * t + 0.9
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            actor_loss, critic_loss = policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            noise.reset()


        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluation = eval_policy(policy, env, args.seed, render=False)
            print(f'evaluating after {t+1} steps')
            mean_return, std_return, min_return, max_return = evaluation
            evaluations.append(evaluation)
            print(mean_return, std_return)
            writer.add_scalar('eval/mean_return', mean_return, t)
            writer.add_scalar('eval/max_return', max_return, t)
            writer.add_scalar('eval/min_return', min_return, t)
            if actor_loss:
                writer.add_scalar('policy/actor-loss', actor_loss, t)
            if critic_loss:
                writer.add_scalar('policy/critic-loss', critic_loss, t)
    #         #debug_critic(policy, args.env, args.seed, gamma = args.discount)
    #         np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
    #         debug_critic(policy, args.env, args.seed, args.discount)
    # save_buffer(replay_buffer, './buffers/hopper_1000k_learning.pkl')

