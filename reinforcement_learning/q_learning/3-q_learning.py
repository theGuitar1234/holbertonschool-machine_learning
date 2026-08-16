#!/usr/bin/env python3
"""Performs Q-learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    When the agent falls in a hole, the reward should be updated to be -1

    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = (Q[state, action] + alpha *
                                 (reward + gamma * np.max(Q[new_state]) -
                                  Q[state, action]))

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
        total_rewards.append(episode_reward)

    return Q, total_rewards
