import numpy as np
import argparse
import gym

from src.env import ToyEnv
from src.viewer import Viewer
from src.policy import RandomPolicy, HardcodedPolicy, TrainedPolicy


def main():

    env = ToyEnv(Viewer=Viewer, ndim=2, num_observables=20)
    env.reset()

    total_reward = 0
    episode = []

    state = None
    policy = HardcodedPolicy(env.state_space, env.action_space)
    for _ in range(10000):

        env.render('human')

        if state is None:
            action = env.action_space.sample()
        else:
            action = policy.get_action(state)

        (reward, state, done) = env.step(action)

        episode.append((state, action, reward))
        total_reward += reward

    print(total_reward)
    env.close()


if __name__ == "__main__":
    main()
