import random
import copy

import numpy as np
import torch
import torch.onnx
from mlagents_envs.environment import ActionTuple
from torch.utils.data import Dataset, DataLoader

from WrapperNet import WrapperNet
from network import QNetwork, action_options, mirrored_actions
from variables import discount, reward_same_action, learning_rate


class Experience:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.predicted_values = []

    def add_instance(self, observations, action, predicted_values, reward):
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predicted_values.append(predicted_values)

    def flip(self):
        new_observations = [(np.flip(vis, 2), nonvis) for vis, nonvis in self.observations]
        new_actions = [mirrored_actions[x] if x is not None else None for x in self.actions]
        new_predicted_values = [np.flip(x, 0) for x in self.predicted_values]

        new_exp = Experience()
        new_exp.observations = new_observations
        new_exp.actions = new_actions
        new_exp.rewards = self.rewards.copy()
        new_exp.predicted_values = new_predicted_values
        return new_exp

    def calculate_targets(self):
        targets = []
        states = []
        for e, observation in enumerate(self.observations):

            if self.actions[e] is None:
                break

            action_index = self.actions[e]
            reward = self.rewards[e]

            if e != 0:
                if self.actions[e] == self.actions[e - 1]:
                    reward += reward_same_action

            # we take the matrix of predicted values and for the actions we had taken adjust the value by the reward
            # and the value of the next state
            target_matrix = self.predicted_values[e].copy()

            # adjust
            target_matrix[action_index] = reward + max(self.predicted_values[e + 1]) * discount
            observation = [arr.astype("float32") for arr in observation]
            target_matrix = target_matrix.astype("float32")
            states.append(observation)
            targets.append(target_matrix)

        return states, targets


class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add_exp(self, exp):
        if not self.is_full():
            self.buffer.append(exp)

    def is_full(self):
        return len(self.buffer) >= self.size

    def create_targets(self):
        state_dataset = []
        targets_dataset = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets()
            targets_dataset += targets
            state_dataset += states
        return state_dataset, targets_dataset

    def flip_dataset(self):
        """
        Mirrors the image and action data in dataset, effectively doubles it.
        :return:
        """
        new_exps = []
        for exp in self.buffer:
            new_exp = exp.flip()
            new_exps.append(new_exp)

        for new_exp in new_exps:
            self.buffer.append(new_exp)

    def wipe(self):
        self.buffer = []


class StateTargetValuesDataset(Dataset):

    def __init__(self, states: list, targets: list):
        self.states = states
        self.targets = targets
        if len(states) != len(targets):
            raise ValueError

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], self.targets[index]


class Trainer:
    def __init__(self, model: QNetwork, buffer_size, device, num_agents=1):
        """
        Class that manages creating a dataset and fitting the model
        :param model:
        :param buffer_size:
        :param device:
        :param num_agents:
        """
        self.device = device

        self.memory = ReplayBuffer(buffer_size)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.num_agents = num_agents

    def train(self, env, exploration_chance):
        """
        Create dataset, fit the model, delete dataset
        :param exploration_chance:
        :param env:
        :return rewards earned:
        """
        # env.reset()
        rewards_stat = self.create_dataset(env, exploration_chance)
        self.memory.flip_dataset()
        self.fit(1)
        self.memory.wipe()
        return rewards_stat

    def create_dataset(self, env, temperature):
        behavior_name = list(env.behavior_specs)[0]
        all_rewards = 0
        # Read and store the Behavior Specs of the Environment
        num_exp = 0

        while not self.memory.is_full():
            num_exp += 1 * self.num_agents
            exps = [Experience() for _ in range(self.num_agents)]
            terminated = [False for _ in range(self.num_agents)]
            while True:
                decision_steps, terminal_steps = env.get_steps(behavior_name)  #
                order = (0, 3, 1, 2)
                decision_steps.obs[0] = np.transpose(decision_steps.obs[0], order)
                terminal_steps.obs[0] = np.transpose(terminal_steps.obs[0], order)

                dis_action_values = []
                cont_action_values = []

                if len(decision_steps) == 0:
                    for agent_id, i in terminal_steps.agent_id_to_index.items():
                        exps[agent_id].add_instance(terminal_steps[agent_id].obs, None,
                                                    np.zeros(self.model.output_shape[1]),
                                                    terminal_steps[agent_id].reward)
                        terminated[agent_id] = True

                else:
                    for agent_id, i in decision_steps.agent_id_to_index.items():

                        if terminated[agent_id]:
                            dis_action_values.append(np.array([0, 0, 0, 0]))
                            cont_action_values.append([])
                            continue
                        # Get the action
                        q_values, action_index = self.model.get_actions(decision_steps[i].obs,temperature)


                        # action_values = action_options[action_index]
                        dis_action_values.append(action_options[action_index][0])
                        cont_action_values.append([])
                        exps[agent_id].add_instance(decision_steps[i].obs, action_index, q_values.copy(),
                                                    decision_steps[i].reward)
                    action_tuple = ActionTuple()
                    final_dis_action_values = np.array(dis_action_values)
                    final_cont_action_values = np.array(cont_action_values)
                    action_tuple.add_discrete(final_dis_action_values)
                    action_tuple.add_continuous(final_cont_action_values)
                    env.set_actions(behavior_name, action_tuple)

                env.step()

                if all(terminated):
                    break
            for exp in exps:
                exp.rewards.pop(0)
                all_rewards += sum(exp.rewards)
                self.memory.add_exp(exp)

        return all_rewards

    def fit(self, epochs: int):
        temp_states, targets = self.memory.create_targets()
        states = []
        for state in temp_states:
            states.append([torch.tensor(obs).to(self.device) for obs in state])
        
        targets = torch.tensor(targets).to(self.device)

        dataset = StateTargetValuesDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        for epoch in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                X, y = batch
                # X = [X[0].view((-1, 1, 64, 64)), X[1].view((-1, 1))]
                # y = y.view(-1, self.model.output_shape[1])

                vis_X = X[0].view((-1, 1, 64, 64))
                nonvis_X = X[1].view((-1, 1))
                X = (vis_X, nonvis_X)

                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                print("loss", loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def save_model(self, path):
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model).cpu()),
            (
                torch.randn((1,) + self.model.visual_input_shape),  # Vis observation
                torch.randn((1,) + self.model.nonvis_input_shape),  # Non vis observation
            ),
            path,
            opset_version=9,
            input_names=['vis_obs', 'nonvis_obs'],
            output_names=['prediction', 'action'],
        )
