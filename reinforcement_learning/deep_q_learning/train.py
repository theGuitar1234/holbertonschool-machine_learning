#!/usr/bin/env python3
"""Train a DQN agent to play Atari's Breakout using keras-rl2."""
import gymnasium as gym
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Permute, Activation
)
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.core import Processor

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Preprocesses Atari frames/rewards for the DQN agent."""

    def process_observation(self, observation):
        """Resize and grayscale a raw Atari frame."""
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img, dtype=np.uint8)
        return processed_observation

    def process_state_batch(self, batch):
        """Normalize a batch of stacked frames to [0, 1]."""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clip rewards to stabilize training."""
        return np.clip(reward, -1., 1.)


class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """Adapts a Gymnasium env to the older step/reset/render API."""

    def reset(self, **kwargs):
        """Return only the observation, dropping the info dict."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Merge terminated/truncated into a single done flag."""
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        """Ignore the mode argument required by keras-rl2's callbacks."""
        return self.env.render()


def build_model(window_length, num_actions):
    """Build the convolutional Q-network used by the DQN agent."""
    input_shape = (window_length,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    return model


def build_agent(model, num_actions):
    """Build the DQN agent with an epsilon-greedy exploration policy."""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
        value_test=.05, nb_steps=1000000)
    dqn = DQNAgent(
        model=model, nb_actions=num_actions, policy=policy, memory=memory,
        processor=processor, nb_steps_warmup=50000, gamma=.99,
        target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])
    return dqn


def main():
    """Train the DQN agent on Breakout and save the final policy network."""
    env = gym.make('ALE/Breakout-v5')
    env = GymnasiumCompatibilityWrapper(env)
    np.random.seed(23)
    env.reset(seed=23)
    num_actions = env.action_space.n

    model = build_model(WINDOW_LENGTH, num_actions)
    dqn = build_agent(model, num_actions)

    dqn.fit(env, nb_steps=1750000, log_interval=10000, visualize=False,
            verbose=2)

    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    main()
