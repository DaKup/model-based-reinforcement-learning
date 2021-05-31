import sys, time
import abc
import datetime
import numpy as np
import gym
import gym_trajectory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_array_ops import Reshape



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

    obs_before = np.dstack([obs[:-1] for (obs, a, r) in episodes])
    obs_after = np.dstack([obs[1:] for (obs, a, r) in episodes])
    a = np.concatenate([a for (obs,a,r) in episodes])
    r = np.concatenate([r for (obs,a,r) in episodes])
    obs_a = np.concatenate((obs_before, a[:,np.newaxis,:]), axis=1)
    transitions = (obs_a, obs_after)
    rewards = (obs_a, r)

    return transitions, rewards



def prepare_datasets_lstm(episodes):

    obs_before = np.stack([obs[:-1] for (obs, a, r) in episodes])
    obs_after = np.stack([obs[:-1] for (obs, a, r) in episodes]) 
    a = np.stack([a for (obs,a,r) in episodes])
    r = np.stack([r for (obs,a,r) in episodes])
    obs_a = np.concatenate((obs_before, a[:,:,np.newaxis,:]), axis=2)
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
    
    num_episodes = 5000
    steps_per_episode = 200

    epochs = 100
    batch_size = 100
    render = False
    
    tensorboard_callback_reward = keras.callbacks.TensorBoard(log_dir="logs/scalars/rewards" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback_transitions = keras.callbacks.TensorBoard(log_dir="logs/scalars/transitions" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    env = gym.make('Trajectory-v0')
    episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)
    # transitions, rewards = prepare_datasets(episodes)
    transitions_lstm, rewards_lstm = prepare_datasets_lstm(episodes)

    validation_episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)
    validation_transitions_lstm, validation_rewards_lstm = prepare_datasets_lstm(validation_episodes)
    
    # transition_net = keras.Sequential([
    #     keras.layers.Input(shape=(3+env.num_observables, env.num_dimensions)),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense((2+env.num_observables)*env.num_dimensions),
    #     keras.layers.Reshape((2+env.num_observables, env.num_dimensions))
    # ])
    # transition_net.compile(optimizer='adam', loss='mse')

    # reward_net = keras.Sequential([
    #     keras.layers.Input(shape=(3+env.num_observables, env.num_dimensions)),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(1)
    # ])
    # reward_net.compile(optimizer='adam', loss='mse')

    transition_net_lstm = keras.Sequential([
        keras.layers.Input(shape=(steps_per_episode, 3+env.num_observables, env.num_dimensions)),
        keras.layers.Reshape((steps_per_episode, (3+env.num_observables) * env.num_dimensions)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense((2+env.num_observables)*env.num_dimensions),
        keras.layers.Reshape((2+env.num_observables, env.num_dimensions))
    ])
    transition_net_lstm.compile(optimizer='adam', loss='mse')

    reward_net_lstm = keras.Sequential([
        keras.layers.Input(shape=(steps_per_episode, 3+env.num_observables, env.num_dimensions)),
        keras.layers.Reshape((steps_per_episode, (3+env.num_observables) * env.num_dimensions)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(1)
    ])
    reward_net_lstm.compile(optimizer='adam', loss='mse')


    transition_net_lstm.fit(transitions_lstm[0], transitions_lstm[1][:, -1], epochs=epochs, batch_size=batch_size, validation_data=(validation_transitions_lstm[0], validation_transitions_lstm[1][:, -1]), callbacks=[tensorboard_callback_transitions])
    reward_net_lstm.fit(rewards_lstm[0], rewards_lstm[1][:, -1], epochs=epochs, batch_size=batch_size, validation_data=(validation_rewards_lstm[0], validation_rewards_lstm[1][:,-1]), callbacks=[tensorboard_callback_reward])



if __name__ == "__main__":
    main()
