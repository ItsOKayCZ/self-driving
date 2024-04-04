from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from Network import QNetwork
from Trainer import Trainer
import KeyboardListener

import os
import datetime
from Variables import MAX_TRAINED_EPOCHS, START_TEMPERATURE, REDUCE_TEMPERATURE, NUM_TRAINING_EXAMPLES,MODEL_PATH
import argparse
import json
import torch

import threading
# for TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-areas', type=int, default=1)
parser.add_argument('-s', '--save-model', action='store_true')
parser.add_argument('-e', '--env', default='./Python Trainer/builds/distance_reward_old_env/selfDriving.x86_64')
parser.add_argument('-D', '--no-display', action='store_true')
parser.add_argument('-t', '--time-scale', type=float, default=1.0)
args = parser.parse_args()
NUM_AREAS = args.num_areas
SAVE_MODEL = args.save_model
ENV_PATH = args.env
NO_DISPLAY = args.no_display
TIME_SCALE = args.time_scale

engine_channel = EngineConfigurationChannel()

listener = KeyboardListener.KeyboardListener()
listener.start()


def launch_tensor_board():
    import os
    os.system('tensorboard --logdir=runs')
    return


def relu(x):
    return max(0.0, x)


if __name__ == "__main__":
    # set up the environment
    env_location = ENV_PATH
    env = UnityEnvironment(file_name=env_location, num_areas=NUM_AREAS, side_channels=[engine_channel])
    engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
    env.reset()

    t = threading.Thread(target=launch_tensor_board, args=([]))
    t.start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'-------- Running on {device}')

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
        qnet = QNetwork(visual_input_shape=(1, 64, 64), nonvis_input_shape=(1,), encoding_size=126, device=device)
        trainer = Trainer(model=qnet, buffer_size=NUM_TRAINING_EXAMPLES, device=device, num_agents=NUM_AREAS,
                          writer=writer)

        if SAVE_MODEL:
            print(f'---- Will save all models to {folder_name} ----')
        else:
            print(f'---- Not saving model as the -s flag is default to "False" ----')

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

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': qnet.state_dict(),
                    'optimizer_state_dict': trainer.optim.state_dict(),
                }, f'{folder_name}/model-epoch-{epoch}.pkl')

                trainer.save_model(f'{folder_name}/model-epoch-{epoch}.onnx')
                listener.reset()

    except KeyboardInterrupt:
        print("\nTraining interrupted, continue to next cell to save to save the model.")

    finally:
        env.close()
        writer.close()
    # Show the training graph
    try:
        if NO_DISPLAY:
            training_data = {
                'num_epochs': len(results),
                'results': results
            }
            with open(f'{folder_name}/training-data.json', 'w') as f:
                json.dump(training_data, f)
            print(f'Saved training data in {folder_name}/training-data.json')
    except ValueError:
        print("\nPlot failed on interrupted training.")
