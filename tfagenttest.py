from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

import gym
import gym_trajectory

# env = suite_gym.load('CartPole-v0')
env = suite_gym.load('Trajectory-v0',
    gym_kwargs={
        'num_dimensions': 2,
        'num_observables': 15,
        'max_targets': 100,
        'max_steps': 5000,
        'max_steps_without_target': 1000,
        'max_position': 100.0,
        'max_acceleration': 10.2,
        'max_velocity': 15.0,
        'collision_epsilon': 10.0
    })
tf_env = tf_py_environment.TFPyEnvironment(env)

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
    
    episode_reward = 0
    episode_steps = 0
    
    # for _ in range(1000):
    while not time_step.is_last():

        action_spec = env.action_spec()
        action = array_spec.sample_bounded_spec(action_spec, np.random.RandomState())
        action = action
        actions = tf.expand_dims(action, axis=0)

        time_step = tf_env.step(actions)

        episode_steps += 1
        episode_reward += time_step.reward.numpy()
        tf_env.render()
    
    rewards.append(episode_reward)
    steps.append(episode_steps)
    time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)

pass

# pip install --upgrade --no-deps --force-reinstall dist\gym_trajectory-0.1-py3-none-any.whl