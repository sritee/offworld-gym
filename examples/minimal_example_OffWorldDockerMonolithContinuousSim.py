#!/usr/bin/env python

# Copyright 2019 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.

import gym
import logging
import time
import numpy as np

from ou_noise import OUNoise

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

# create the environment
from offworld_gym.envs.common.channels import Channels
env = gym.make("OffWorldDockerMonolithContinuousSim-v0", channel_type=Channels.DEPTH_ONLY)
env.seed(42)

logging.info(f"action space: {env.action_space} observation_space: {env.observation_space}")
ep = 0
cnt = 0
ou_noise = OUNoise(env.action_space.shape[0])
while ep < 50:
    env.reset()
    ou_noise.reset()
    done = False
    while not done:
        #sampled_action = env.action_space.sample()
        sampled_action = ou_noise.noise()
        np.clip(sampled_action, -0.5, 0.5)
        #sampled_action[0] = 0
        #env.render()
        obs, rew, done, info = env.step(sampled_action)
        downsampled = obs[0, ::4, ::4, 0]
        plt.imshow(downsampled)
        plt.show()
        #plt.imshow(obs[0, :, :, 0])
        #plt.show()
        time.sleep(0.05)
        
    if rew:
        cnt +=1
    ep += 1    
print(cnt, ep)