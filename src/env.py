import numpy as np
import matplotlib.pyplot as plt
import gym



class ToyEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['console', 'file', 'human']}

    MAX_POSITION    = 100
    MAX_VELOCITY    = 1
    ACCELERATION    = 1
    TARGET_EPSILON  = 0.1
    MAX_REWARD      = 10

    def __init__(self, ndim=2, num_observables=5):
        super(ToyEnv, self).__init__()

        self.ndim = ndim
        self.num_observables = num_observables

        self.action_space = gym.spaces.MultiDiscrete([
            3 for _ in range(self.ndim) # backward, still, forward
            ])
        
        self.state_space = gym.spaces.Tuple([
            # position:
            gym.spaces.Box(
                low=-self.MAX_POSITION,
                high=self.MAX_POSITION,
                shape=(self.ndim,), dtype=np.float32),
            
            # velocity:
            gym.spaces.Box(
                low=-self.MAX_VELOCITY,
                high=self.MAX_VELOCITY,
                shape=(self.ndim,), dtype=np.float32),
            
            # observable targets:
            gym.spaces.Tuple([
                gym.spaces.Box(
                    low=-self.MAX_POSITION,
                    high=self.MAX_POSITION,
                    shape=(self.ndim,), dtype=np.float32) for _ in range(self.num_observables)
            ])
        ])


        # observable:
        self.agent_position     = None
        self.agent_velocity     = None
        self.target_positions   = None

        # latent:
        self.previous_distance  = None

        # visualization:
        self.fig = plt.figure()

    
    def reset(self):
        
        (
            self.agent_position,
            self.agent_velocity,
            self.target_positions
        ) = self.state_space.sample()

        self.previous_distance = np.linalg.norm(self.agent_position - self.target_positions[0])

    
    def step(self, action):
        
        self.agent_velocity += (action-1) * self.ACCELERATION
        self.agent_velocity = np.clip(
            self.agent_velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY)

        self.agent_position += self.agent_velocity
        self.agent_position = np.clip(
            self.agent_position, -self.MAX_POSITION, self.MAX_POSITION)

        distance = np.linalg.norm(self.agent_position - self.target_positions[0])
        if distance < self.TARGET_EPSILON:
            reward = MAX_REWARD
            self.target_positions = (*self.target_positions[1:-1], self.state_space[0].sample())
            self.previous_distance = np.linalg.norm(self.agent_position - self.target_positions[0])
        else:
            reward = self.previous_distance - distance
            self.previous_distance = distance

        return (
            reward,
            (
                self.agent_position,
                self.agent_velocity,
                self.target_positions
            )
        )


    def render(self, mode='console'):
        
        assert(self.ndim == 2)
        if mode == 'human':
            if self.fig is None:
                self.fig = plt.figure()
                plt.ion()
                plt.show()
            
            plt.scatter(self.agent_position[0], self.agent_position[1], c='black')
            plt.scatter(self.target_positions[0][0], self.target_positions[0][1], c='red')

            self.fig.gca().set_xlim(-self.MAX_POSITION, self.MAX_POSITION)
            self.fig.gca().set_ylim(-self.MAX_POSITION, self.MAX_POSITION)
            plt.draw()
            plt.pause(0.0000001)
            plt.cla()
            # plt.show(block=False)
            # show agent black
            # show observable targets blue-gray
            # highlight current target green
            # # highlight captured targets?

            # # # other idea, agent can pick his next target

        elif mode == 'console':
            print(self.previous_distance)
            # print stats:
            # current reward
            # distance to next target
            # number of captured targets
            # current position, velocity, action
            pass
        elif mode == 'file':
            pass

# test = ToyEnv()
# pass

# class GoLeftEnv(gym.Env):
#   """
#   Custom Environment that follows gym interface.
#   This is a simple env where the agent must learn to go always left. 
#   """
#   # Because of google colab, we cannot implement the GUI ('human' render mode)
#   metadata = {'render.modes': ['console']}
#   # Define constants for clearer code
#   LEFT = 0
#   RIGHT = 1

#   def __init__(self, grid_size=10):
#     super(GoLeftEnv, self).__init__()

#     # Size of the 1D-grid
#     self.grid_size = grid_size
#     # Initialize the agent at the right of the grid
#     self.agent_pos = grid_size - 1

#     # Define action and observation space
#     # They must be gym.spaces objects
#     # Example when using discrete actions, we have two: left and right
#     n_actions = 2
#     self.action_space = spaces.Discrete(n_actions)
#     # The observation will be the coordinate of the agent
#     # this can be described both by Discrete and Box space
#     self.observation_space = spaces.Box(low=0, high=self.grid_size,
#                                         shape=(1,), dtype=np.float32)

#   def reset(self):
#     """
#     Important: the observation must be a numpy array
#     :return: (np.array) 
#     """
#     # Initialize the agent at the right of the grid
#     self.agent_pos = self.grid_size - 1
#     # here we convert to float32 to make it more general (in case we want to use continuous actions)
#     return np.array([self.agent_pos]).astype(np.float32)

#   def step(self, action):
#     if action == self.LEFT:
#       self.agent_pos -= 1
#     elif action == self.RIGHT:
#       self.agent_pos += 1
#     else:
#       raise ValueError("Received invalid action={} which is not part of the action space".format(action))

#     # Account for the boundaries of the grid
#     self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

#     # Are we at the left of the grid?
#     done = bool(self.agent_pos == 0)

#     # Null reward everywhere except when reaching the goal (left of the grid)
#     reward = 1 if self.agent_pos == 0 else 0

#     # Optionally we can pass additional info, we are not using that for now
#     info = {}

#     return np.array([self.agent_pos]).astype(np.float32), reward, done, info

#   def render(self, mode='console'):
#     if mode != 'console':
#       raise NotImplementedError()
#     # agent is represented as a cross, rest as a dot
#     print("." * self.agent_pos, end="")
#     print("x", end="")
#     print("." * (self.grid_size - self.agent_pos))

#   def close(self):
#     pass


# class ToyEnvironment:


#     class ActionSpace:

#         def __init__(self):
#                 super().__init__()
    

#     class StateSpace:

#         def __init__(self):
#                 super().__init__()


#     def __init__(
#         self,
#         ndim = 2,
#         T = 500,
#         M = 250,
#         a_min = -100,
#         a_max = 100,
#         d_min = 1,
#         d_max = 300,
#         v_min = -500,
#         v_max = 500,
#         num_observables = 3,
#         max_agent_distance = 30000,
#         max_distance_from_origin = 1500,
#         epochs = 50,
#         batch_size = 32
#         ):
#         super().__init__()

#         # self.ndim                = 2             # number of dimensions
#         # self.T                   = 500           # maximum time steps and length of each trajectory
#         # self.M                   = 250           # number of generated trajectories using pi_explore

#         # self.a_max               = 100           # maximum acceleration
#         # self.a_min               = -a_max        # minimum acceleration

#         # self.d_max                      = 3 * a_max     # max distance between trajectory points
#         # self.d_min                      = 1             # min distance between trajectory points

#         # self.v_max                      = 5 * a_max     # maximum velocity
#         # self.v_min                      = -v_max        # minimum velocity

#         # self.num_observables            = 3             # number of observable targets from the current timestep
#         # self.max_agent_distance         = 100 * d_max   # maximum distance an agent may have to his next target before the episode is canceled
#         # self.max_distance_from_origin   = T * d_max     # use as normalization factor

#         # self.epochs                     = 50
#         # self.batch_size                 = 32

#         self.ndim = ndim
#         self.T = T
#         self.M = M
#         self.a_min = a_min
#         self.a_max = a_max
#         self.d_min = d_min
#         self.d_max = d_max
#         self.v_min = v_min
#         self.v_max = v_max
#         self.num_observables = num_observables
#         self.max_agent_distance = max_agent_distance
#         self.max_distance_from_origin = max_distance_from_origin
#         self.epochs = epochs
#         self.batch_size = batch_size
    
    
