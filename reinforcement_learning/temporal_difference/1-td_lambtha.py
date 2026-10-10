#!/usr/bin/env python3
"""TD(lambda) algorithm for value estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(lambda) algorithm.

    env is the environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next
        action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros(V.shape[0])

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)

            eligibility[state] += 1
            delta = reward + gamma * V[next_state] - V[state]
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
