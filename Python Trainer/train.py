import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from network import QNetwork
from trainer import Trainer
from scheduler import Scheduler
import os
import datetime
from variables import max_trained_epochs,exploration_chance_start,exploration_reduce,num_training_examples
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

if __name__ == "__main__":
	# set up the environment
	# env_location = './env/Self driving.exe'
	env_location = ENV_PATH
	env = UnityEnvironment(file_name=env_location, num_areas=NUM_AREAS)
	env.reset()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'----- Running on {device}')

	# get the action space and observation space
	print(env.behavior_specs)
	behavior_name = list(env.behavior_specs)[0]
	print(f"Name of the behavior : {behavior_name}")
	spec = env.behavior_specs[behavior_name]
	observation_shape = spec.observation_specs
	print(observation_shape)
	num_actions = spec.action_spec
	print(num_actions)

	num_epochs = max_trained_epochs
	# expl_chance = exploration_chance_start
	# expl_reduce = exploration_reduce

	results = []
	try:
		qnet = QNetwork(visual_input_shape = (1, 64, 64), nonvis_input_shape=(1,), encoding_size=126, device=device)
		trainer = Trainer(model=qnet,buffer_size=num_training_examples, device=device, num_agents=NUM_AREAS)
		scheduler = Scheduler(num_epochs, exploration_chance_start)

		if SAVE_MODEL:
			folder_name = f'./models/{datetime.datetime.now().strftime("%y-%m-%d %H%M%S")}'
			os.makedirs(folder_name)
			print(f'---- Will save models into {folder_name}')
		else:
			print(f'---- Not saving model as the -s flag is default to "False"')

		for epoch in range(num_epochs):
			expl_chance = scheduler.exploration
			print(f"epoch: {epoch}, exploration chance:{expl_chance}")
			reward = trainer.train(env, expl_chance)
			reward /= NUM_AREAS
			results.append(reward)
			scheduler.step()
			# expl_chance *= expl_reduce

			if SAVE_MODEL:
				trainer.save_model(f'{folder_name}/model-epoch-{epoch}.onnx')

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
