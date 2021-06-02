import numpy as np

# from dearpygui.core import start_dearpygui
# from dearpygui.demo import show_demo

from dearpygui import core, simple


def collect_data(
    env,
    num_episodes,
    num_steps_per_episode,
    policy,
    render=False
):

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


def main():
    
    # env = gym.make('Trajectory-v0')
    # episodes = collect_data(env, num_episodes, steps_per_episode, RandomPolicy(env), render)

    # show_demo()

    def save_callback(sender, data):
        print("Save Clicked")

    with simple.window("Generate Dataset"):
        core.add_text("Hello world")
        core.add_button("Save", callback=save_callback)
        core.add_input_text("string")
        core.add_slider_float("float")

    core.start_dearpygui()

if __name__ == "__main__":
    main()
