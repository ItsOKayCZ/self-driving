import argparse
import datetime
import json
import os
import threading

import torch
from data_channel import DataChannel
from keyboard_listener import KeyboardListener
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from network import QNetwork

# for TensorBoard
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
from variables import (
    ENCODING_SIZE,
    MAX_TRAINED_EPOCHS,
    MODEL_PATH,
    NONVISUAL_INPUT_SHAPE,
    NUM_TRAINING_EXAMPLES,
    REDUCE_TEMPERATURE,
    START_TEMPERATURE,
    VISUAL_INPUT_SHAPE,
)

writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-areas", type=int, default=1)
parser.add_argument("-s", "--save-model", action="store_true")
parser.add_argument("-e", "--env", default="./build/selfDriving.x86_64")
parser.add_argument("-D", "--no-display", action="store_true")
parser.add_argument("-t", "--time-scale", type=float, default=1.0)
parser.add_argument("-i", "--interactive", action="store_true")
args = parser.parse_args()
NUM_AREAS = args.num_areas
SAVE_MODEL = args.save_model
ENV_PATH = args.env
NO_DISPLAY = args.no_display
TIME_SCALE = args.time_scale
INTERACTIVE = args.interactive

engine_channel = EngineConfigurationChannel()
data_channel = DataChannel()

listener = KeyboardListener()
listener.start()


def launch_tensor_board():
    import os

    os.system("tensorboard --logdir=runs")
    return


def relu(x):
    return max(0.0, x)


if __name__ == "__main__":
    # set up the environment
    env_location = ENV_PATH
    if INTERACTIVE:
        env_location = None
        print("Waiting for unity environment")
    env = UnityEnvironment(
        file_name=env_location, num_areas=NUM_AREAS, side_channels=[engine_channel, data_channel]
    )

    # Wide - 15
    # Slim - 10
    data_channel.set_int_parameter("roadSize", 15)
    # 0 -> Amazon road
    # 1 -> Black & white road
    data_channel.set_int_parameter("roadColor", 0)
    data_channel.set_int_parameter("cameraWidth", 64)
    data_channel.set_int_parameter("cameraHeight", 64)

    engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
    env.reset()

    t = threading.Thread(target=launch_tensor_board, args=([]))
    t.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-------- Running on {device}")

    # get the action space and observation space
    print(env.behavior_specs)
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    observation_shape = spec.observation_specs
    print(observation_shape)
    num_actions = spec.action_spec
    print(num_actions)

    num_epochs = MAX_TRAINED_EPOCHS
    temperature = START_TEMPERATURE
    temperature_red = REDUCE_TEMPERATURE

    folder_name = f'{MODEL_PATH}/{datetime.datetime.now().strftime("%y-%m-%d %H%M%S")}'

    results = []
    try:
        qnet = QNetwork(
            visual_input_shape=VISUAL_INPUT_SHAPE,
            nonvis_input_shape=NONVISUAL_INPUT_SHAPE,
            encoding_size=ENCODING_SIZE,
            device=device,
        )
        trainer = Trainer(
            model=qnet,
            buffer_size=NUM_TRAINING_EXAMPLES,
            device=device,
            num_agents=NUM_AREAS,
            writer=writer,
        )

        if SAVE_MODEL:
            print(f"---- Will save all models to {folder_name} ----")
        else:
            print('---- Not saving model as the -s flag is default to "False" ----')

        for epoch in range(num_epochs):
            print("------Training------")
            print(f"Epoch: {epoch}, Temperature:{temperature}")
            reward = trainer.train(env, temperature)

            print("------Done------")
            reward /= NUM_TRAINING_EXAMPLES
            print(f"Reward earned: {reward}")

            temperature = relu(temperature - temperature_red)
            writer.add_scalar("Reward/Train", reward, epoch)
            writer.flush()

            if SAVE_MODEL or listener.was_pressed():
                if not os.path.isdir(folder_name):
                    os.makedirs(folder_name)

                torch.save(qnet, f"{folder_name}/model-epoch-{epoch}.pkl")
                trainer.save_model(f"{folder_name}/model-epoch-{epoch}.onnx")
                listener.reset()

    except KeyboardInterrupt:
        print("\nTraining interrupted, continue to next cell to save to save the model.")

    finally:
        env.close()
        writer.close()
    # Show the training graph
    try:
        if NO_DISPLAY:
            training_data = {"num_epochs": len(results), "results": results}
            with open(f"{folder_name}/training-data.json", "w") as f:
                json.dump(training_data, f)
            print(f"Saved training data in {folder_name}/training-data.json")
    except ValueError:
        print("\nPlot failed on interrupted training.")
