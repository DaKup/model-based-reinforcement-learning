import numpy as np
import argparse
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from src.env import ToyEnv
from src.viewer import Viewer
from src.policy import RandomPolicy, HardcodedPolicy, TrainedPolicy


def main():

    env = ToyEnv(Viewer=Viewer, ndim=2, num_observables=5)
    env.reset()

    nb_actions = env.action_space.n()
    # nb_actions = 1000
    nb_observations = (10, 20, 30)

    # Next, we build a very simple model.

    input_shape = (1,) + nb_observations
    output_shape = nb_actions


    model = Sequential()
    model.add(Flatten(input_shape))

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
    # even the metrics!
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=SequentialMemory(limit=50000, window_length=1),
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=BoltzmannQPolicy()
        )
    
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
    dqn.save_weights('dqn_{}_weights.h5f'.format('trajectory'), overwrite=True)
    dqn.test(env, nb_episodes=5, visualize=True)


    # total_reward = 0
    # episode = []

    # state = None
    # policy = HardcodedPolicy(env.state_space, env.action_space)
    # for _ in range(10000):

    #     env.render('human')

    #     if state is None:
    #         action = env.action_space.sample()
    #     else:
    #         action = policy.get_action(state)

    #     (reward, state, done) = env.step(action)

    #     episode.append((state, action, reward))
    #     total_reward += reward

    # print(total_reward)
    # env.close()


if __name__ == "__main__":
    main()
