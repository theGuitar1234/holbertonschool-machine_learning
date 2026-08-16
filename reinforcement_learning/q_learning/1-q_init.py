#!/usr/bin/env python3
"""Initializes the Q-table"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    env is the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray of zeros
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    return np.zeros((state_size, action_size))
