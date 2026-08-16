#!/usr/bin/env python3
"""Uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
