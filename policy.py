import abc


class BasePolicy:

    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def sample(self, obs):
        return



class RandomPolicy(BasePolicy):

    def __init__(self, env):
        self.env = env

    def sample(self, obs):
        return self.env.action_space.sample()
