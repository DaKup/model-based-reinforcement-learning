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
from tf_agents.replay_buffers import replay_buffer
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.drivers.py_driver import PyDriver
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy

# from tf_agents.metrics.py_metrics import AverageReturnMetric
from tf_agents.metrics.tf_metrics import AverageReturnMetric

from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.agents.random.random_agent import RandomAgent
from tf_agents.agents.ppo.ppo_policy import PPOPolicy

tf.compat.v1.enable_v2_behavior()

# tf.enable_eager_execution()

import gym
import gym_trajectory

def main():

    env = suite_gym.load('Trajectory-v0',
        gym_kwargs={
            'num_dimensions': 2,
            'num_observables': 15,
            'max_targets': 100,
            'max_steps': 5000,
            'max_steps_without_target': 5000,
            'max_position': 100.0,
            'max_acceleration': 10.2,
            'max_velocity': 15.0,
            'collision_epsilon': 10.0
        })
    tf_env = tf_py_environment.TFPyEnvironment(env)

    agent = RandomAgent(time_step_spec=tf_env.time_step_spec(), action_spec=tf_env.action_spec())

    metric = AverageReturnMetric()
    replay_buffer = []
    # uniform_replay_buffer = PyUniformReplayBuffer(data_spec=agent.collect_data_spec, capacity=2000)
    uniform_replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=1)
    # observers = [replay_buffer.append, metric]

    # driver = PyDriver(
    #     env,
    #     policy=RandomPyPolicy(env.time_step_spec(), env.action_spec()),
    #     observers=[replay_buffer.append, metric],
    #     max_steps=2000
    # )

    # driver = TFDriver(
    #     tf_env,
    #     # policy=RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec()),
    #     policy=agent.policy,
    #     observers=[uniform_replay_buffer],
    #     max_steps=2000
    # )

    driver = DynamicStepDriver(
        tf_env,
        policy=agent.policy,
        observers=[uniform_replay_buffer.add_batch],#, metric],
        # transition_observers=None,
        num_steps=1000)

    agent.initialize()
    initial_time_step = tf_env.reset()
    final_time_step, final_policy_state = driver.run(initial_time_step)

    dataset = uniform_replay_buffer.as_dataset()

    # print('Replay Buffer:')
    # for traj in replay_buffer:
    #   print(traj)

    # agent = TFAgent()

    # print('Average Return: ', metric.result())



if __name__ == "__main__":
    main()
