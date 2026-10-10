#!/usr/bin/env python3
"""SARSA(lambda) algorithm."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose an action using the epsilon-greedy policy.

    Q is the Q table
    state is the current state
    epsilon is the threshold for exploration
    Returns: the index of the chosen action
    """
    p = np.random.uniform()
    if p > epsilon:
        return int(np.argmax(Q[state]))
    return int(np.random.randint(0, Q.shape[1]))


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm.

    env is the environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    init_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action
            if done:
                break

        epsilon = (min_epsilon + (init_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q
