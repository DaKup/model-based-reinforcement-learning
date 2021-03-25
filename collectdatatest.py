import sys, time
import numpy as np
import gym
import gym_trajectory
import tensorflow as tf
from tensorflow import keras

def flatten(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item

def main():

    env = gym.make('Trajectory-v0')

    actions = []
    rewards = []
    observations = [tf.reshape(tf.convert_to_tensor(list(flatten(env.reset()))), [-1])]

    for i in range(100):
        a = env.action_space.sample()
        (obs, r, done, info) = env.step(a)

        a = list(flatten(a))
        obs = list(flatten(obs))

        observations.append(tf.reshape(tf.convert_to_tensor(obs), [-1]))
        actions.append(a)
        rewards.append(r)

    observations = tf.convert_to_tensor(observations)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)

    observations_actions = tf.concat([observations[:-1], actions], axis=1)
    dataset1 = (observations_actions, rewards)
    dataset2 = (observations_actions, observations[1:])
    pass

    # obs, a => r
    # obs, a => obs
    # obs, a, a, a, ... => r


if __name__ == "__main__":
    main()
