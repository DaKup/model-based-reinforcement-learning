import numpy as np



class Policy:


    def __init__(self, state_space, action_space):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space


    def get_action(self, state):

        RuntimeError("not implemented")



class RandomPolicy(Policy):


    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)
    

    def get_action(self, state=None):

        return self.action_space.sample()



class HardcodedPolicy(Policy):


    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)
    

    def get_action(self, state):

        agent_position = state[0]
        target_position = state[2][0]
        action = np.ones_like(self.action_space.sample())
        action[np.where(agent_position > target_position)] = 0
        action[np.where(agent_position < target_position)] = 2
        return action



class TrainedPolicy(Policy):


    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)


    def get_action(self, state):
        pass
