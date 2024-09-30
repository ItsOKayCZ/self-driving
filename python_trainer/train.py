import argparse
import datetime
import json
from pathlib import Path

import torch
from data_channel import DataChannel
from keyboard_listener import KeyboardListener
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from network import QNetwork
from tensorboard import program

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

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-areas", type=int, default=1)
parser.add_argument("-s", "--save-model", action="store_true")
parser.add_argument(
    "-e",
    "--env",
    default=str(Path(__file__).parent / "build" / "StandaloneLinux64" / "selfDriving"),
)
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


def launch_tensor_board(logs_location: Path) -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logs_location)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def print_env_info(env: UnityEnvironment) -> None:
    # get the action space and observation space
    print(env.behavior_specs)  # noqa: T201
    behavior_name = next(iter(env.behavior_specs))
    print(f"Name of the behavior : {behavior_name}")  # noqa: T201
    spec = env.behavior_specs[behavior_name]
    observation_shape = spec.observation_specs
    print(observation_shape)  # noqa: T201
    num_actions = spec.action_spec
    print(num_actions)  # noqa: T201


if __name__ == "__main__":
    # Start TensorBoard
    log_location = Path(__file__).parent / "runs"
    writer = SummaryWriter(log_location / datetime.datetime.now().strftime("%y-%m-%d %H%M%S"))
    launch_tensor_board(log_location)

    # Start keyboard listener for saving
    listener = KeyboardListener()
    listener.start()

    # Set up the environment
    engine_channel = EngineConfigurationChannel()
    data_channel = DataChannel()
    env_location = ENV_PATH
    if INTERACTIVE:
        env_location = None
        print("Waiting for unity environment")  # noqa: T201
    env = UnityEnvironment(
        file_name=env_location,
        num_areas=NUM_AREAS,
        side_channels=[engine_channel, data_channel],
    )
    # Wide - 15
    # Slim - 10
    data_channel.set_int_parameter("roadSize", 15)
    # 0 -> Amazon road
    # 1 -> Black & white road
    data_channel.set_int_parameter("roadColor", 0)
    engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")  # noqa: T201
    print_env_info(env)

    num_epochs = MAX_TRAINED_EPOCHS
    temperature = START_TEMPERATURE
    temperature_red = REDUCE_TEMPERATURE

    model_folder = Path(MODEL_PATH) / datetime.datetime.now().strftime("%y-%m-%d %H%M%S")

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
            print(f"---- Will save all models to {model_folder} ----")  # noqa: T201
        else:
            print('---- Not saving model as the -s flag is default to "False" ----')  # noqa: T201

        for epoch in range(num_epochs):
            print("------Training------")  # noqa: T201
            print(f"Epoch: {epoch}, Temperature:{temperature}")  # noqa: T201
            reward = trainer.train(env, temperature)

            print("------Done------")  # noqa: T201
            reward /= NUM_TRAINING_EXAMPLES
            print(f"Reward earned: {reward}")  # noqa: T201

            temperature = max(0.0, temperature - temperature_red)
            writer.add_scalar("Reward/Train", reward, epoch)
            writer.flush()

            if SAVE_MODEL or listener.was_pressed():
                folder = Path(model_folder)
                folder.mkdir(parents=True, exist_ok=True)

                torch.save(qnet, model_folder / f"model-epoch-{epoch}.pkl")
                trainer.save_model(model_folder / f"model-epoch-{epoch}.onnx")
                listener.reset()

    except KeyboardInterrupt:
        print("\nTraining interrupted, continue to next cell to save to save the model.")  # noqa: T201

    finally:
        env.close()
        writer.close()
    # Show the training graph
    try:
        if NO_DISPLAY:
            training_data = {"num_epochs": len(results), "results": results}
            with (model_folder / "training-data.json").open("w") as f:
                json.dump(training_data, f)
            print(f"Saved training data in {model_folder}/training-data.json")  # noqa: T201
    except ValueError:
        print("\nPlot failed on interrupted training.")  # noqa: T201
