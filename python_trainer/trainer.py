import copy

import numpy as np
import torch
import torch.onnx
from buffer import Experience, ReplayBuffer, StateTargetValuesDataset
from mlagents_envs.environment import ActionTuple
from network import QNetwork, action_options
from torch.utils.data import DataLoader
from tqdm import tqdm
from variables import LEARNING_RATE
from wrapper_net import WrapperNet


class Trainer:
    def __init__(self, model: QNetwork, buffer_size, device, num_agents=1, writer=None):
        """
        Class that manages creating a dataset and fitting the model
        :param model:
        :param buffer_size:
        :param device:
        :param num_agents:
        """
        self.device = device
        self.writer = writer

        self.curr_epoch = 0

        self.memory = ReplayBuffer(buffer_size)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)

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
        sample_exp = self.memory.buffer[int(self.memory.size() / 2)]
        sample_image = sample_exp.observations[int(len(sample_exp) / 2)][0]
        sample_q_values = sample_exp.predicted_values[int(len(sample_exp) / 2)]

        self.writer.add_image("Sample image", sample_image)

        # Add text
        steer = ""

        for s in sample_q_values:
            steer += f"{s:.2f} "
        self.writer.add_text("Sample Q values (steer)", steer, self.curr_epoch)

        self.fit(1)
        self.memory.wipe()
        return rewards_stat

    def create_dataset(self, env, temperature):
        behavior_name = list(env.behavior_specs)[0]
        all_rewards = 0
        # Read and store the Behavior Specs of the Environment

        exps = [Experience() for _ in range(self.num_agents)]
        # terminated = [False for _ in range(self.num_agents)]
        env.reset()
        bar = tqdm(total=self.memory.max_size)

        while len(self.memory) + sum([len(x) for x in exps]) < self.memory.max_size:
            decision_steps, terminal_steps = env.get_steps(behavior_name)  #

            dis_action_values = []
            cont_action_values = []

            if len(decision_steps) == 0:
                for agent_id, i in terminal_steps.agent_id_to_index.items():
                    exp = exps[agent_id]
                    exp.add_instance(
                        terminal_steps[agent_id].obs,
                        None,
                        np.zeros(self.model.output_shape[1]),
                        terminal_steps[agent_id].reward,
                    )
                    exp.rewards.pop(0)
                    bar.update()
                    all_rewards += sum(exp.rewards)
                    self.memory.add_exp(exp)
                    exps[agent_id] = Experience()

            else:
                for agent_id, i in decision_steps.agent_id_to_index.items():
                    # Get the action
                    q_values, action_index = self.model.get_actions(
                        decision_steps[i].obs, temperature
                    )

                    dis_action_values.append(action_options[action_index][0])
                    cont_action_values.append([])
                    exps[agent_id].add_instance(
                        decision_steps[i].obs,
                        action_index,
                        q_values.copy(),
                        decision_steps[i].reward,
                    )
                    bar.update()

                action_tuple = ActionTuple()
                final_dis_action_values = np.array(dis_action_values)
                final_cont_action_values = np.array(cont_action_values)
                action_tuple.add_discrete(final_dis_action_values)
                action_tuple.add_continuous(final_cont_action_values)
                env.set_actions(behavior_name, action_tuple)

            env.step()

        for exp in exps:
            if len(exp.actions) == 0:
                continue
            exp.actions[-1] = None
            self.memory.add_exp(exp)
            all_rewards += sum(exp.rewards)

        return all_rewards

    def fit(self, epochs: int):
        temp_states, targets = self.memory.create_targets()
        states = []
        for state in temp_states:
            states.append([torch.tensor(obs).to(self.device) for obs in state])

        targets = torch.tensor(targets).to(self.device)

        dataset = StateTargetValuesDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        loss_sum = 0
        count = 0

        for _ in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                x, y = batch

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                print("loss", loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1

        self.writer.add_scalar("Loss/Epoch", loss_sum / count, self.curr_epoch)
        self.curr_epoch += 1

    def save_model(self, path):
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model).cpu()),
            (
                # Vis observation
                torch.randn((1,) + self.model.visual_input_shape),
                # Non vis observation
                torch.randn((1,) + self.model.nonvis_input_shape),
            ),
            path,
            opset_version=9,
            input_names=["vis_obs", "nonvis_obs"],
            output_names=["prediction", "action"],
        )
