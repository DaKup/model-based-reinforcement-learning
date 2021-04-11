import sys, time
import abc
import numpy as np
import gym
import gym_trajectory
import tensorflow as tf
from tensorflow import keras



class BasePolicy:

    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def sample(obs):
        return


class RandomPolicy(BasePolicy):

    def __init__(self, env):
        self.env = env

    def sample(self, obs):
        return self.env.action_space.sample()



def collect_data(env, num_episodes, num_steps_per_episode, policy, render=False):

    episodes = []
    for _ in range(num_episodes):
        
        obs = env.reset()
        np_obs = np.zeros(shape=(2+env.num_observables, env.num_dimensions))
        np_obs[0] = obs[0]
        np_obs[1] = obs[1]
        np_obs[2:] = obs[2]
        observations = [np_obs]
        if render:
            env.render()
        actions = []
        rewards = []
        for _ in range(num_steps_per_episode):
            
            a = policy.sample(observations[-1])
            (obs, r, done, info) = env.step(a)
            if render:
                env.render()
            np_obs = np.zeros(shape=(2+env.num_observables, env.num_dimensions))
            np_obs[0] = obs[0]
            np_obs[1] = obs[1]
            np_obs[2:] = obs[2]
            observations.append(np_obs)
            actions.append(np.array(a))
            rewards.append(np.array(r))

        episodes.append((np.array(observations), np.array(actions), np.array(rewards)))
    return episodes



def prepare_datasets(episodes):

    obs_before = np.concatenate([obs[:-1] for (obs, a, r) in episodes])
    obs_after = np.concatenate([obs[1:] for (obs, a, r) in episodes])
    a = np.concatenate([a for (obs,a,r) in episodes])
    r = np.concatenate([r for (obs,a,r) in episodes])
    obs_a = np.concatenate((obs_before, a[:,np.newaxis,:]), axis=1)
    transitions = (obs_a, obs_after)
    rewards = (obs_a, r)

    return transitions, rewards


def main():
    
    env = gym.make('Trajectory-v0')
    episodes = collect_data(env, 10, 100, RandomPolicy(env))
    transitions, rewards = prepare_datasets(episodes)
    
    transition_net = keras.Sequential([
        keras.layers.Input(shape=(3+env.num_observables, env.num_dimensions)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense((2+env.num_observables)*env.num_dimensions),
        keras.layers.Reshape((2+env.num_observables, env.num_dimensions))
    ])
    transition_net.compile(optimizer='adam', loss='mse')
    transition_net.fit(transitions[0], transitions[1], epochs=15, batch_size=10)
    
    
    reward_net = keras.Sequential([
        keras.layers.Input(shape=(3+env.num_observables, env.num_dimensions)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    reward_net.compile(optimizer='adam', loss='mse')
    reward_net.fit(rewards[0], rewards[1], epochs=15, batch_size=10)
    pass



if __name__ == "__main__":
    main()
