#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import random
train = __import__('train').train

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

env = gym.make('CartPole-v1')
set_seed(env, 0)

scores = train(env, 20)
env.close()
