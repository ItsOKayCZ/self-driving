import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from network import QNetwork
from trainer import Trainer
import torch
import os

if __name__ == "__main__":
    # set up the environment
    # env_location = "C:\\Users\\tasek\\Desktop\\programovani\\staz\\auticka\\self-driving\\Python Trainer\\env\\Self driving.exe"
    env_location = './env/Self driving.exe'
    env = UnityEnvironment(file_name=env_location)
    env.reset()

    # get the action space and observation space
    print(env.behavior_specs)
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]

    num_epochs = 70
    observation_shape = spec.observation_specs
    print(observation_shape)
    num_actions = spec.action_spec
    print(num_actions)

    try:
        qnet = QNetwork((1, 64, 64), 3, 126, 4)
        trainer = Trainer(model=qnet,buffer_size=1000)
        results = []
        for epoch in range(num_epochs):
          print(epoch)
          reward = trainer.train(env)
          results.append(reward)

    except KeyboardInterrupt:
        print("\nTraining interrupted, continue to next cell to save to save the model.")

    finally:
        env.close()

    # Show the training graph
    try:
        plt.plot(range(num_epochs), results)
    except ValueError:
        print("\nPlot failed on interrupted training.")