#!/usr/bin/env python3
"""Display a game of Atari's Breakout played by a trained DQN agent."""
import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import (
    AtariProcessor, GymnasiumCompatibilityWrapper, WINDOW_LENGTH,
    build_model
)


def build_agent(model, num_actions):
    """Build a DQN agent that always exploits its learned policy."""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model, nb_actions=num_actions, policy=policy, memory=memory,
        processor=processor)
    dqn.compile('adam', metrics=['mae'])
    return dqn


def main():
    """Load the trained policy network and play one episode visually."""
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = GymnasiumCompatibilityWrapper(env)
    num_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, num_actions)
    dqn = build_agent(model, num_actions)
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
