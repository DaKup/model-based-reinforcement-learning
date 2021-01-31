from __future__ import absolute_import, division, print_function
import base64
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_agents

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

from tf_agents.networks.q_network import QNetwork

from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.utils.common import element_wise_squared_loss
from tf_agents.trajectories.trajectory import from_transition

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

tf.compat.v1.enable_v2_behavior()

import gym_trajectory
env = suite_gym.load('Trajectory-v0')
# env = suite_gym.load('CartPole-v1')

env = tf_py_environment.TFPyEnvironment(env)

# q_net = QNetwork(env.observation_spec(), env.action_spec())
train_step_counter = tf.Variable(0)

# agent = DqnAgent(env.time_step_spec(),
#                 env.action_spec(),
#                 q_network=q_net,
#                 optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
#                 td_errors_loss_fn=element_wise_squared_loss,
#                 train_step_counter=train_step_counter)

# agent = PPOAgent(env.time_step_spec(),
#                 env.action_spec()
#                 actor_net = None,
#                 optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
#                 td_errors_loss_fn=element_wise_squared_loss,
#                 train_step_counter=train_step_counter)

def create_networks(observation_spec, action_spec):

    preprocessing_combiner = tf.keras.layers.Concatenate()
    actor_net = ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(16, 8, 4), (32, 4, 2)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        preprocessing_combiner=preprocessing_combiner,
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu)
    value_net = ValueRnnNetwork(
        observation_spec,
        conv_layer_params=[(16, 8, 4), (32, 4, 2)],
        input_fc_layer_params=(256,),
        preprocessing_combiner=preprocessing_combiner,
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu)

    return actor_net, value_net

actor_net, value_net = create_networks(env.observation_spec(), env.action_spec())

global_step = tf.compat.v1.train.get_or_create_global_step()
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-5)

num_epochs = 100

tf_agent = PPOAgent(
    env.time_step_spec(),
    env.action_spec(),
    optimizer,
    actor_net,
    value_net,
    num_epochs=num_epochs,
    train_step_counter=global_step,
    discount_factor=0.995,
    gradient_clipping=0.5,
    entropy_regularization=1e-2,
    importance_ratio_clipping=0.2,
    use_gae=True,
    use_td_lambda_return=True
)

agent.initialize()

replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                    batch_size=env.batch_size,
                    max_length=100000)

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(env, agent.policy, 5)
returns = [avg_return]



collect_steps_per_iteration = 1
batch_size = 64
dataset = replay_buffer.as_dataset(num_parallel_calls=3, 
                                    sample_batch_size=batch_size, 
                                    num_steps=2).prefetch(3)
iterator = iter(dataset)
num_iterations = 10000
env.reset()

for _ in range(batch_size):
    collect_step(env, agent.policy, replay_buffer)

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    # Print loss every 200 steps.
    if step % 200 == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    # Evaluate agent's performance every 1000 steps.
    if step % 1000 == 0:
        avg_return = compute_avg_return(env, agent.policy, 5)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


plt.figure(figsize=(12,8))
iterations = range(0, num_iterations + 1, 1000)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.show()