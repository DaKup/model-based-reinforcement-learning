import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.style as style
# style.use('classic')


class ToyRender:


    def __init__(
        self,
        env):
        super().__init__()

        self.env = env

        self.fig, self.axs = plt.subplots(
            ncols=2, nrows=2,
            # num=style_label,
            # figsize=fig_size,
            # title='ToyEnv',
            constrained_layout=True)
            # squeeze=True)
        
        self.fig.canvas.set_window_title('Toy Environment')
        # self.fig.suptitle('Toy Environment')

        self.velocity = np.zeros(shape=(100, 2))

        # plt.ion()


    def render(self):
            
        self.axs[0, 0].cla()
        self.axs[0, 0].set_title('arena')

        self.axs[0, 0].scatter(self.env.agent_position[0], self.env.agent_position[1], c='black')
        self.axs[0, 0].scatter(self.env.target_positions[:][0], self.env.target_positions[:][1], c='gray')
        self.axs[0, 0].scatter(self.env.target_positions[0][0], self.env.target_positions[0][1], c='green')

        self.velocity = np.roll(self.velocity, -1, axis=0)
        self.velocity[-1] = self.env.agent_velocity

        self.axs[1, 0].cla()
        self.axs[1, 0].set_title('velocity')

        self.axs[1, 0].plot(self.velocity[:, 0], c='b')
        self.axs[1, 0].plot(self.velocity[:, 1], c='r')

        self.axs[0, 0].set_xlim(-self.env.MAX_POSITION, self.env.MAX_POSITION)
        self.axs[0, 0].set_ylim(-self.env.MAX_POSITION, self.env.MAX_POSITION)

        self.axs[1, 0].set_xlim(0, self.velocity.shape[0])
        self.axs[1, 0].set_ylim(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY)

        plt.draw()
        plt.pause(0.0000001)

        # highlight current target green
        # # highlight captured targets?

        # # # other idea, agent can pick his next target
