#!/usr/bin/env python3
"""Has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    You should always exploit the Q-table

    Returns: total_rewards, rendered_outputs
        total_rewards is the total rewards for the episode
        rendered_outputs is a list of rendered outputs representing
            the board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0.0
    rendered_outputs = []

    rendered_outputs.append(_format_render(env.render()))

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        rendered_outputs.append(_format_render(env.render()))
        total_rewards += reward

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs


def _format_render(output):
    """Replaces ANSI highlight codes with double quotes"""
    output = output.replace('\x1b[41m', '"').replace('\x1b[0m', '"')
    return output.strip('\n').lstrip(' ')
