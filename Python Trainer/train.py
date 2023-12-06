import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from network import QNetwork
from trainer import Trainer
import os
import datetime
from variables import (MAX_TRAINED_EPOCHS,
                       START_TEMPERATURE,
                       REDUCE_TEMPERATURE,
                       NUM_TRAINING_EXAMPLES,
                       IMAGE_SHAPE,
                       ENCODING_SIZE,
                       LEARNING_RATE,
                       NUM_EVALUATION_EXAMPLES
                       )
import argparse
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-areas', type=int, default=1)
parser.add_argument('-s', '--save-model', action='store_true')
parser.add_argument('-e', '--env', default='./env/Self driving.exe')
parser.add_argument('-D', '--no-display', action='store_true')
args = parser.parse_args()
NUM_AREAS = args.num_areas
SAVE_MODEL = args.save_model
ENV_PATH = args.env
NO_DISPLAY = args.no_display


def relu(x):
    return max(0.0, x)


if __name__ == "__main__":
    # set up the environment
    # env_location = './env/Self driving.exe'
    env_location = ENV_PATH
    env = UnityEnvironment(file_name=env_location, num_areas=NUM_AREAS)
    env.reset()

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

    results = []
    try:
        qnet = QNetwork(visual_input_shape=IMAGE_SHAPE, nonvis_input_shape=(1,), encoding_size=ENCODING_SIZE, device=device)
        trainer = Trainer(model=qnet, buffer_size=NUM_TRAINING_EXAMPLES, device=device, learning_rate=LEARNING_RATE,num_evaluations=NUM_EVALUATION_EXAMPLES,num_agents=NUM_AREAS)

        if SAVE_MODEL:
            folder_name = f'./models/{datetime.datetime.now().strftime("%y-%m-%d %H%M%S")}'
            os.makedirs(folder_name)
            print(f'---- Will save models into {folder_name}')
        else:
            print(f'---- Not saving model as the -s flag is default to "False"')

        for epoch in range(num_epochs):
            print(f"epoch: {epoch}, temperature:{temperature}")
            reward = trainer.train(env, temperature)
            reward /= NUM_AREAS
            reward /= NUM_TRAINING_EXAMPLES
            results.append(reward)
            temperature = relu(temperature - temperature_red)

            if SAVE_MODEL:
                trainer.save_model(f'{folder_name}/model-epoch-{epoch}-reward-{reward}.onnx')

            print(f"reward earned: {reward}")

    except KeyboardInterrupt:
        print("\nTraining interrupted, continue to next cell to save to save the model.")

    finally:
        env.close()

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
        else:
            plt.plot(range(len(results)), results)
            plt.show()
    except ValueError:
        print("\nPlot failed on interrupted training.")
