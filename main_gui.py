from dearpygui.core import *
from dearpygui.simple import *

# import gym
# import gym_trajectory

# env = gym.make('Trajectory-v0')
# observation = env.reset()

# while True:
#     (observation, reward, done, info) = env.step(env.action_space.sample())
#     env.render()

def theme_callback(sender, data):
    set_theme(sender)


def main():

    set_main_window_title("Model-Based Reinforcement Learning")

    with window("Model"):
        
        set_window_pos("Model", 50, 110)
        with menu("Themes"):
            add_menu_item("Dark", callback=theme_callback)
            add_menu_item("Light", callback=theme_callback)
            add_menu_item("Classic", callback=theme_callback)
            add_menu_item("Dark 2", callback=theme_callback)
            add_menu_item("Grey", callback=theme_callback)
            add_menu_item("Dark Grey", callback=theme_callback)
            add_menu_item("Cherry", callback=theme_callback)
            add_menu_item("Purple", callback=theme_callback)
            add_menu_item("Gold", callback=theme_callback)
            add_menu_item("Red", callback=theme_callback)

        
        add_label_text("label_text", default_value="default_value")

        # set_value("label_text", "new value")
        # add_texture("texture", data, width, height, format=0)
        # add_button("Save", callback=save_callback)
        # lstm vs fcn
        # state model, reward model
        # current model // not saved yet, or path to file
        # load model
        # save model // overwrite warning
        pass

    with window("Observations"):

        set_window_pos("Observations", 200, 200)
        # load dataset
        # generate more data
        # save dataset
        # shuffle
        # train/validation sets
        pass

    with window("Training"):
        
        set_window_pos("Training", 520, 220)
        # epochs, batch size
        # start training
        # start validate
        pass

    with window("Inference"):

        set_window_pos("Inference", 1050, 20)
        # start inference
        # popup window rendering
        # save to video file
        pass

    start_dearpygui(primary_window="Model")


if __name__ == "__main__":
    main()
