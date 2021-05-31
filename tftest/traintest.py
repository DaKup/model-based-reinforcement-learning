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



def predict_observation(env, net, obs, a):

    input_batch = np.zeros(shape=(1, 3+env.num_observables, env.num_dimensions))
    input_batch[0, 0] = obs[0]
    input_batch[0, 1] = obs[1]
    input_batch[0, 2:-1] = obs[2]
    input_batch[0, -1] = a

    output_batch = net(input_batch)
    output_batch = output_batch.numpy()

    obs_pred = (output_batch[0, 0], output_batch[0, 1], tuple([_ for _ in output_batch[0, 2:]]))
    return obs_pred


def predict_reward(env, net, obs, a):
    
    input_batch = np.zeros(shape=(1, 3+env.num_observables, env.num_dimensions))
    input_batch[0, 0] = obs[0]
    input_batch[0, 1] = obs[1]
    input_batch[0, 2:-1] = obs[2]
    input_batch[0, -1] = a

    output_batch = net(input_batch)
    output_batch = output_batch.numpy()

    r_pred = output_batch[0].item()
    return r_pred



def main():
    
    num_episodes = 1000
    steps_per_episode = 500

    epochs = 100
    batch_size = 500
    render = False


    env = gym.make('Trajectory-v0')
    episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)
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

    reward_net = keras.Sequential([
        keras.layers.Input(shape=(3+env.num_observables, env.num_dimensions)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    reward_net.compile(optimizer='adam', loss='mse')
    
    transition_net.evaluate(transitions[0], transitions[1], batch_size=batch_size)
    reward_net.evaluate(rewards[0], rewards[1], batch_size=batch_size)
    
    
    transition_net.fit(transitions[0], transitions[1], epochs=epochs, batch_size=batch_size)
    reward_net.fit(rewards[0], rewards[1], epochs=epochs, batch_size=batch_size)
    
    
    # 2nd iteration training:
    env = gym.make('Trajectory-v0')
    episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)
    transitions, rewards = prepare_datasets(episodes)
    transition_net.fit(transitions[0], transitions[1], epochs=epochs, batch_size=batch_size)
    reward_net.fit(rewards[0], rewards[1], epochs=epochs, batch_size=batch_size)

    # evaluate:
    env = gym.make('Trajectory-v0')
    episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)
    transitions, rewards = prepare_datasets(episodes)
    transition_net.evaluate(transitions[0], transitions[1], batch_size=batch_size)
    reward_net.evaluate(rewards[0], rewards[1], batch_size=batch_size)

    
    # test inference:
    env = gym.make('Trajectory-v0')
    obs = env.reset()
    policy = RandomPolicy(env)
    a = policy.sample(obs)

    # predicted obs and r:
    obs_pred = predict_observation(env, transition_net, obs, a)
    r_pred = predict_reward(env, reward_net, obs, a)

    # actual obs and r:
    (obs_real, r_real, done, info) = env.step(a)

    pass

if __name__ == "__main__":
    main()
