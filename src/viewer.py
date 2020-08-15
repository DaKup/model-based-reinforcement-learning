import numpy
from gym.envs.classic_control import rendering
from src.env import ToyEnv



class Viewer:

    VIEWER_WIDTH = 600
    VIEWER_HEIGHT = 400

    AGENT_COLOR = (1, 0, 0)
    AGENT_SIZE = 5

    TARGET_COLOR = (0, 1, 0)
    TARGET_SIZE = 5

    OBSERVABLE_COLOR = (0, 0, 1)
    OBSERVABLE_SIZE = 3

    def __init__(
        self,
        env: ToyEnv
        ):
        super().__init__()

        self.env = env
        self.viewer = None

        self.offset = (self.env.MAX_POSITION, self.env.MAX_POSITION)
        self.scale = (self.VIEWER_WIDTH / (2*self.env.MAX_POSITION), self.VIEWER_HEIGHT / (2*self.env.MAX_POSITION))

        self.agent = None
        self.agent_transform = None

        self.target = None
        self.target_transform = None

        self.observable = []
        self.observable_transform = []


    def render(self, mode):
        
        if self.viewer is None:
            
            # viewer:
            self.viewer = rendering.Viewer(self.VIEWER_WIDTH, self.VIEWER_HEIGHT)
            
            # agent:
            self.agent_transform = rendering.Transform()
            self.agent = rendering.make_circle(self.AGENT_SIZE)
            self.agent.set_color(*self.AGENT_COLOR)
            self.agent.add_attr(self.agent_transform)
            self.viewer.add_geom(self.agent)

            # target:
            self.target_transform = rendering.Transform()
            self.target = rendering.make_circle(self.TARGET_SIZE)
            self.target.set_color(*self.TARGET_COLOR)
            self.target.add_attr(self.target_transform)
            self.viewer.add_geom(self.target)

            # observable targets:
            for i in range(self.env.num_observables - 1):
                observable_transform = rendering.Transform()
                observable = rendering.make_circle(self.OBSERVABLE_SIZE)
                observable.set_color(*self.OBSERVABLE_COLOR)
                observable.add_attr(observable_transform)
                self.viewer.add_geom(observable)
                self.observable.append(observable)
                self.observable_transform.append(observable_transform)

        self.agent_transform.set_translation(*(self.scale * (self.env.agent_position + self.offset)))
        self.target_transform.set_translation(*(self.scale * (self.env.target_positions[0] + self.offset)))
        
        for idx, observable_transform in enumerate(self.observable_transform):
            self.observable[idx].set_color(0, 1 - idx * 1/len(self.observable_transform), idx * 1/len(self.observable_transform))
            observable_transform.set_translation(*(self.scale * (self.env.target_positions[idx+1] + self.offset)))

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')
