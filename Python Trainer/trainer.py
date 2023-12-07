import random
import copy

import numpy as np
import torch
import torch.onnx
from mlagents_envs.environment import ActionTuple
from torch.utils.data import Dataset, DataLoader

from WrapperNet import WrapperNet
from network import QNetwork
from variables import LEARNING_RATE
from Buffer import ReplayBuffer, Experience, StateTargetValuesDataset


class Trainer:
    def __init__(self, model: QNetwork, buffer_size, device, learning_rate, num_evaluations, num_agents=1):
        """
        Class that manages creating a dataset and fitting the model

        :param model:
        :param buffer_size:
        :param device:
        :param learning_rate:
        :param num_evaluations:
        :param num_agents:
        """
        self.device = device

        self.memory = ReplayBuffer(buffer_size)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.num_agents = num_agents

        # number of tries per model in evaluation

        self.num_evaluations = num_evaluations

    def train(self, env, exploration_chance) -> int:
        """
        Create dataset, fit the model, delete dataset
        :param exploration_chance:
        :param env:
        :return rewards earned:
        """
        # env.reset()
        rewards_stat = self.create_dataset(env, exploration_chance)
        self.memory.flip_dataset()
        new_model = self.fit(2)
        if self.evaluate(env,new_model):
            self.model = new_model
        self.memory.wipe()
        return rewards_stat

    def create_dataset(self, env, temperature):
        behavior_name = list(env.behavior_specs)[0]
        all_rewards = 0
        # Read and store the Behavior Specs of the Environment
        num_exp = 0
        plotting = True;  # turn on plotting of q values and probs
        toPlot = False

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
                        exps[agent_id].add_instance(terminal_steps[agent_id].obs,
                                                    None,
                                                    (np.zeros(self.model.output_shape_speed),
                                                     np.zeros(self.model.output_shape_steer)),
                                                    terminal_steps[agent_id].reward)
                        terminated[agent_id] = True

                else:
                    for agent_id, i in decision_steps.agent_id_to_index.items():

                        if terminated[agent_id]:
                            dis_action_values.append(np.array([]))
                            cont_action_values.append([0, 0])
                            continue
                        # Get the action

                        q_values, actions, indices = self.model.get_actions(decision_steps[i].obs, temperature,
                                                                            toPlot=toPlot)
                        # action_values = action_options[action_index]

                        dis_action_values.append([])
                        cont_action_values.append(actions)

                        ######
                        if len(exps[0].actions) and plotting == 40:
                            toPlot = True
                        else:
                            toPlot = False
                        ########
                        exps[agent_id].add_instance(decision_steps[i].obs,
                                                    indices,
                                                    (q_values[0].detach().cpu().numpy(),
                                                     q_values[1].detach().cpu().numpy()),
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

    def fit(self, epochs: int) -> QNetwork:

        new_model = copy.deepcopy(self.model)

        temp_states, targets = self.memory.create_targets()
        states = []
        targets_speed = targets[0]
        targets_steer = targets[1]
        for state in temp_states:
            states.append([torch.tensor(obs).to(self.device) for obs in state])

        targets_speed = torch.tensor(np.array(targets_speed)).to(self.device)
        targets_steer = torch.tensor(np.array(targets_steer)).to(self.device)

        dataset = StateTargetValuesDataset(states, targets_speed, targets_steer)
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

                y_hat = new_model(X)
                loss = self.loss_fn(y_hat[0], y[0]) + self.loss_fn(y_hat[1], y[1])
                print("loss", loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        return new_model

    def evaluate(self, env, new_model: QNetwork) -> bool:
        print("------Evaluating------")
        old_model_scores = [0] * self.num_evaluations
        new_model_scores = [0] * self.num_evaluations
        print("Testing new model...")
        self.fill_scores(env, new_model, new_model_scores)
        print("Testing old model...")
        self.fill_scores(env, self.model, old_model_scores)
        new_model_fitness_score = sum(new_model_scores)/self.num_evaluations
        old_model_fitness_score = sum(old_model_scores)/self.num_evaluations
        print(f"New model score: {new_model_fitness_score}, Old model score: {old_model_fitness_score}")
        if new_model_fitness_score >= old_model_fitness_score:
            print("Updating model...")
            return True
        else:
            print("Not updating model...")
            return False

    def fill_scores(self, env, model: QNetwork, scores_list: list):

        behavior_name = list(env.behavior_specs)[0]
        next_index_to_fill = 0
        terminated = [False] * self.num_evaluations
        while not all(terminated):

            env.reset() # for new tracks
            while True:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                order = (0, 3, 1, 2)
                decision_steps.obs[0] = np.transpose(decision_steps.obs[0], order)
                terminal_steps.obs[0] = np.transpose(terminal_steps.obs[0], order)

                dis_action_values = []
                cont_action_values = []

                if len(decision_steps) == 0:
                    for agent_id, i in terminal_steps.agent_id_to_index.items():

                        scores_list[agent_id + next_index_to_fill] += terminal_steps[agent_id].reward
                        terminated[agent_id+next_index_to_fill] = True

                else:

                    for agent_id, i in decision_steps.agent_id_to_index.items():
                        if agent_id + next_index_to_fill >= len(scores_list) or terminated[agent_id+next_index_to_fill]:
                            dis_action_values.append(np.array([]))
                            cont_action_values.append([0, 0])
                            continue
                        # Get the action
                        q_values, actions, indices = model.get_actions(decision_steps[i].obs,0.5 )
                        scores_list[agent_id + next_index_to_fill] += decision_steps[i].reward
                        dis_action_values.append([])
                        cont_action_values.append(actions)

                    action_tuple = ActionTuple()
                    final_dis_action_values = np.array(dis_action_values)
                    final_cont_action_values = np.array(cont_action_values)
                    action_tuple.add_discrete(final_dis_action_values)
                    action_tuple.add_continuous(final_cont_action_values)
                    env.set_actions(behavior_name, action_tuple)
                env.step()
                if all(terminated) or all(terminated[i] for i in range(next_index_to_fill,next_index_to_fill+self.num_agents)):
                    break

            next_index_to_fill += self.num_agents

    def save_model(self, path):
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model).cpu()),
            (
                torch.randn((1,) + self.model.visual_input_shape),  # Vis observation
                torch.randn((1,) + self.model.nonvis_input_shape),  # Non vis observation
            ),
            path,
            opset_version=11,
            input_names=['vis_obs', 'nonvis_obs'],
            output_names=['prediction', 'action'],
        )
