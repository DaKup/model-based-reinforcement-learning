import numpy as np
import gym



class ToyEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['console', 'file', 'human']}

    MAX_POSITION    = 100
    MAX_VELOCITY    = 2
    ACCELERATION    = 0.5
    TARGET_EPSILON  = 15
    MAX_REWARD      = 10

    def __init__(self, ndim=2, num_observables=5, Viewer=None):
        super(ToyEnv, self).__init__()

        # self.action_space
        # self.observation_space
        # self.reward_range

        self.ndim = ndim
        self.num_observables = num_observables

        self.action_space = gym.spaces.MultiDiscrete([
            3 for _ in range(self.ndim) # backward, still, forward
            ])
        
        self.observation_space = gym.spaces.Tuple([
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
        self.Viewer = Viewer
        self.viewer = None

    
    def reset(self):
        
        (
            self.agent_position,
            self.agent_velocity,
            self.target_positions
        ) = self.observation_space.sample()

        self.previous_distance = np.linalg.norm(self.agent_position - self.target_positions[0])
        self.viewer = None

    
    def step(self, action):
        
        self.agent_velocity += (action-1) * self.ACCELERATION
        self.agent_velocity = np.clip(
            self.agent_velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY)

        self.agent_position += self.agent_velocity
        self.agent_position = np.clip(
            self.agent_position, -self.MAX_POSITION, self.MAX_POSITION)

        distance = np.linalg.norm(self.agent_position - self.target_positions[0])
        if distance < self.TARGET_EPSILON:
            reward = self.MAX_REWARD
            self.target_positions = (*self.target_positions[1:], self.observation_space[0].sample())
            self.previous_distance = np.linalg.norm(self.agent_position - self.target_positions[0])
        else:
            reward = self.previous_distance - distance
            self.previous_distance = distance

        done = False
        return (
            reward,
            (
                self.agent_position,
                self.agent_velocity,
                self.target_positions
            ),
            done
        )


    def render(self, mode='human'):

        if mode == 'human':
            assert(self.ndim == 2)
            if self.viewer is None:
                self.viewer = self.Viewer(self)
            return self.viewer.render(mode)

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


    def close(self):
        pass


    def seed(self):
        pass
