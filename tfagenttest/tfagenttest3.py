from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tensorflow import keras
# from tensorflow.keras import layers

# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
# from tf_agents.replay_buffers import replay_buffer
# from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
# from tf_agents.trajectories import time_step as ts

# from tf_agents.drivers.py_driver import PyDriver
# from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

# from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.fixed_policy import FixedPolicy

# from tf_agents.metrics.py_metrics import AverageReturnMetric
from tf_agents.metrics.tf_metrics import AverageReturnMetric

# from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.agents.random.random_agent import RandomAgent
# from tf_agents.agents.ppo.ppo_policy import PPOPolicy

from tf_agents.networks.network import Network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.sequential import Sequential

from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent

from tf_agents.utils.common import element_wise_squared_loss


tf.compat.v1.enable_v2_behavior()

import gym
import gym_trajectory

def main():

    env = suite_gym.load('Trajectory-v0', gym_kwargs={
        'num_dimensions': 2,
        'num_observables': 3,
        'max_targets': 100,
        'max_steps': 5000,
        'max_steps_without_target': 5000,
        'max_position': 100.0,
        'max_acceleration': 10.2,
        'max_velocity': 15.0,
        'collision_epsilon': 10.0
    })
    tf_env = tf_py_environment.TFPyEnvironment(env)

    agent = RandomAgent(tf_env.time_step_spec(), tf_env.action_spec())
    uniform_replay_buffer = TFUniformReplayBuffer(agent.collect_data_spec, batch_size=1)

    transitions = []

    driver = DynamicStepDriver(
        tf_env,
        policy=agent.policy,
        observers=[uniform_replay_buffer.add_batch],
        transition_observers=[transitions.append],
        num_steps=500
    )

    initial_time_step = tf_env.reset()
    final_time_step, final_policy_state = driver.run(initial_time_step)
    dataset = uniform_replay_buffer.as_dataset()

    input_state = []
    input_action = []
    output_state = []
    output_reward = []
    for transition in transitions:
        input_state.append(tf.concat(tf.nest.flatten(transition[0].observation), axis=-1))
        input_action.append(tf.concat(tf.nest.flatten(transition[1].action), axis=-1))
        output_state.append(tf.concat(tf.nest.flatten(transition[2].observation), axis=-1))
        output_reward.append(tf.concat(tf.nest.flatten(transition[2].reward), axis=-1))

    tf_input_state = tf.squeeze(tf.stack(input_state), axis=1)
    tf_input_action = tf.squeeze(tf.stack(input_action), axis=1)
    tf_output_state = tf.squeeze(tf.stack(output_state), axis=1)
    tf_output_reward = tf.stack(output_reward)
     
    # dataset = (features, labels)

    # (time_step_before, policy_step_action, time_step_after) = transitions[0]
    # observation = time_step_before.observation
    # action = policy_step_action.action
    # # (discount_, observation_, reward_, step_type_) = time_step_after
    # observation_ = time_step_after.observation

    pass

    # dataset = tf.data.Dataset.from_tensors([sample1, sample2, sample3])
    # dataset = tf.data.Dataset.from_tensors(transitions)
    # create ReplayBuffer() or MapDataset()

    # input_tensor = observation, action
    # output_tensor = observation_

    # for iter in dataset.as_numpy_iterator():
        # pass

    # input_shape = tf_env.time_step_spec() + tf_env.action_spec()
    # output_size = tf_env.time_step_spec()
    # transition_net = keras.Sequential([
    #     keras.layers.Dense(32, input_shape=input_shape),
    #     keras.layers.Activation('relu'),
    #     keras.layers.Dense(32),
    #     keras.layers.Activation('relu'),
    #     keras.layers.Dense(output_size)#,
    #     # keras.layers.Activation('tanh')
    # ])
    # transition_net.compile(optimizer='adam', loss='mse')


    # q_net = QNetwork(input_tensor_spec=tf_env.observation_spec(), action_spec=tf_env.action_spec(), fc_layer_params=(100,))
    # # optimizer = tf.compat.v1.train.AdamOptimizer()
    # optimizer = keras.optimizers.Adam()
    # train_step_counter = tf.compat.v2.Variable(0)

    # tf_agent = DqnAgent(
    #     tf_env.time_step_spec(),
    #     tf_env.action_spec(),
    #     q_network=q_net,
    #     optimizer=optimizer,
    #     td_errors_loss_fn = element_wise_squared_loss,
    #     train_step_counter=train_step_counter)

    # tf_agent.initialize()


if __name__ == "__main__":
    main()
