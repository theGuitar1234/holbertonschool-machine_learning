#!/usr/bin/env python3
"""Play a FrozenLake episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode by always choosing the highest-valued action."""
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
