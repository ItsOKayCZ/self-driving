import random
import copy

import numpy as np
import torch
import torch.onnx
from mlagents_envs.environment import ActionTuple
from torch.utils.data import Dataset, DataLoader
from variables import IMAGE_SHAPE, START_TEMPERATURE

from WrapperNet import WrapperNet
from network import QNetwork
from Buffer import ReplayBuffer, Experience, StateTargetValuesDataset


class Trainer:
    def __init__(self, model: QNetwork, buffer_size, device, learning_rate, num_evaluations, num_agents=1, writer=None):
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
        self.writer = writer
        # number of tries per model in evaluation

        self.curr_epoch = 0
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
        sample_image = self.memory.buffer[0].observations[-1][0].reshape(IMAGE_SHAPE)

        sample_q_values = self.memory.buffer[0].predicted_values[-1]
        self.writer.add_image("Sample image", sample_image)

        self.writer.add_histogram("Sample Q values (steer)", sample_q_values[1])
        self.writer.add_histogram("Sample Q values (speed)", sample_q_values[0])
        self.memory.flip_dataset()
        new_model = self.fit(2)
        if self.evaluate(env, new_model, exploration_chance):
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

        exps = [Experience() for _ in range(self.num_agents)]
        terminated = [False for _ in range(self.num_agents)]
        n_active_agents = self.num_agents
        while not self.memory.is_full():

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            dis_action_values = []
            cont_action_values = []

            if len(decision_steps) == 0:
                for agent_id, i in terminal_steps.agent_id_to_index.items():
                    exp = exps[agent_id]
                    exp.add_instance(terminal_steps[agent_id].obs,
                                     None,
                                     (np.zeros(self.model.output_shape_speed),
                                      np.zeros(self.model.output_shape_steer)),
                                     terminal_steps[agent_id].reward)
                    exp.rewards.pop(0)
                    all_rewards += sum(exp.rewards)
                    self.memory.add_exp(exp)
                    print(f"{len(self.memory) * 100 / self.memory.size}%")
                    if len(self.memory) + n_active_agents > self.memory.size:
                        n_active_agents -= 0
                        terminated[agent_id] = True
                    else:
                        exps[agent_id] = Experience()

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
        return all_rewards

    def fit(self, epochs: int) -> QNetwork:

        print("Fitting...")
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
        loss_sum = 0
        count = 0
        for epoch in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                X, y = batch
                # X = [X[0].view((-1, 1, 64, 64)), X[1].view((-1, 1))]
                # y = y.view(-1, self.model.output_shape[1])

                vis_X = X[0].view((-1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
                nonvis_X = X[1].view((-1, 1))
                X = (vis_X, nonvis_X)

                y_hat = new_model(X)
                loss = self.loss_fn(y_hat[0], y[0]) + self.loss_fn(y_hat[1], y[1])
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1

        self.writer.add_scalar("Loss/Epoch", loss_sum / count, self.curr_epoch)
        self.curr_epoch += 1
        return new_model

    def evaluate(self, env, new_model: QNetwork, temperature: float) -> bool:
        print("------ Evaluating ------")

        print("Testing new model...")
        new_model_scores = self.fill_scores(env, new_model)

        print("Testing old model...")
        old_model_scores = self.fill_scores(env, self.model)

        new_model_fitness_score = sum(new_model_scores) / self.num_evaluations
        old_model_fitness_score = sum(old_model_scores) / self.num_evaluations

        print(f"New model score: {new_model_fitness_score}, Old model score: {old_model_fitness_score}")
        if new_model_fitness_score >= old_model_fitness_score:
            print("Updating model...")
            return True
        else:
            if np.random.uniform() < temperature / START_TEMPERATURE:
                print("Overriding, updating model...")
                return True
            print("Not updating model...")
            return False

    def fill_scores(self, env, model: QNetwork) -> list:
        score_list = []
        scores = [0 for _ in range(self.num_agents)]
        n_active_agents = self.num_agents
        terminated = [False for _ in range(self.num_agents)]
        behavior_name = list(env.behavior_specs)[0]

        while len(score_list) != self.num_evaluations:

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            dis_action_values = []
            cont_action_values = []

            if len(decision_steps) == 0:
                for agent_id, i in terminal_steps.agent_id_to_index.items():
                    score_list.append(scores[agent_id])
                    if n_active_agents + len(score_list) > self.num_evaluations:
                        n_active_agents -= 1
                        terminated[agent_id] = True
                    else:
                        scores[agent_id] = 0
                    print(f"{len(score_list) * 100 / self.num_evaluations}%")
            else:

                for agent_id, i in decision_steps.agent_id_to_index.items():
                    if terminated[agent_id]:
                        dis_action_values.append(np.array([]))
                        cont_action_values.append([0, 0])
                        continue
                    # Get the action

                    q_values, actions, indices = model.get_actions(decision_steps[i].obs, 0.5)
                    scores[agent_id] += decision_steps[i].reward
                    dis_action_values.append([])
                    cont_action_values.append(actions)

                action_tuple = ActionTuple()
                final_dis_action_values = np.array(dis_action_values)
                final_cont_action_values = np.array(cont_action_values)
                action_tuple.add_discrete(final_dis_action_values)
                action_tuple.add_continuous(final_cont_action_values)
                env.set_actions(behavior_name, action_tuple)

            env.step()

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
