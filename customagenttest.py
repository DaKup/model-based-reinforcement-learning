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