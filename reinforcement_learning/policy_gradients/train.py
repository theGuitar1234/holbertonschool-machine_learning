#!/usr/bin/env python3
"""Full training with Monte-Carlo policy gradient."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implements a full training.

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    show_result: whether to render the environment every 1000 episodes
    Returns: all values of the score
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0
        done = False

        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            grads.append(grad)
            rewards.append(reward)
            score += reward

        for i in range(len(grads)):
            G = sum([r * gamma ** t for t, r in enumerate(rewards[i:])])
            weight += alpha * grads[i] * G

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
