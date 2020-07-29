import argparse
import gym
from src.env import ToyEnv
import numpy as np

def main():

    env = ToyEnv()
    env.reset()

    total_reward = 0
    episode = []
    for _ in range(10000):
        env.render('human')
        action = env.action_space.sample()
        state = env.state_space.sample()

        # manual policy:
        agent_position = state[0]
        target_position = state[2][0]
        action = np.ones_like(action)
        action[np.where(agent_position > target_position)] = 0
        action[np.where(agent_position < target_position)] = 2

        (reward, state, done) = env.step(action)
        episode.append((state, action, reward))
        total_reward += reward
    env.close()

    print(total_reward)


if __name__ == "__main__":
    main()
