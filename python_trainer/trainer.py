import copy

import numpy as np
import torch
import torch.onnx
from buffer import Experience, ReplayBuffer, StateTargetValuesDataset
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from network import QNetwork, action_options
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from variables import (
    BLUR_INTENSITY,
    LEARNING_RATE,
    NOISE_INTESITY,
    NOISE_OPACITY,
    REWARD_MAX,
    VISUAL_INPUT_SHAPE,
)
from wrapper_net import WrapperNet


class Trainer:
    def __init__(
        self,
        model: QNetwork,
        buffer_size: int,
        device: torch.device,
        num_agents: int = 1,
        writer: SummaryWriter | None = None,
    ) -> None:
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
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-7,
        )

        self.num_agents = num_agents

    def train(self, env: UnityEnvironment, temperature: float) -> float:
        """
        Create dataset, fit the model, delete dataset
        :param exploration_chance:
        :param env:
        :return rewards earned:
        """

        rewards_stat = self.create_dataset(env, temperature)
        self.memory.flip_dataset()
        sample_exp = self.memory.buffer[int(self.memory.size() / 2)]
        sample_image = sample_exp.observations[int(len(sample_exp) / 2)][0]
        sample_q_values = sample_exp.predicted_values[int(len(sample_exp) / 2)]

        if self.writer is not None:
            self.writer.add_image("Sample image", sample_image)

            # Turn to True, if you want to see all of the images from the epoch

            # Add text
            steer = ""

            for s in sample_q_values:
                steer += f"{s:.2f} "

            self.writer.add_text("Sample Q values (steer)", steer, self.curr_epoch)

        self.fit(1)
        self.memory.wipe()
        return rewards_stat

    def create_dataset(self, env: UnityEnvironment, temperature: float) -> float:
        behavior_name = next(iter(env.behavior_specs))
        all_rewards = 0
        # Read and store the Behavior Specs of the Environment

        exps = [Experience() for _ in range(self.num_agents)]

        env.reset()
        bar = tqdm(total=self.memory.max_size)

        while len(self.memory) + sum([len(x) for x in exps]) < self.memory.max_size:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            dis_action_values = []
            cont_action_values = []

            if len(decision_steps) == 0:
                for agent_id in terminal_steps:
                    exp = exps[agent_id]
                    state_obs, reward = self.get_state_and_reward(terminal_steps[agent_id])
                    exp.add_instance(
                        state_obs,
                        None,
                        np.zeros(self.model.output_shape[1]),
                        reward,
                    )
                    exp.rewards.pop(0)
                    bar.update()
                    all_rewards += sum(exp.rewards)
                    self.memory.add_exp(exp)
                    exps[agent_id] = Experience()

            else:
                for agent_id in decision_steps:
                    state_obs, reward = self.get_state_and_reward(decision_steps[agent_id])

                    # Get the action
                    q_values, action_index = self.model.get_actions(
                        state_obs,
                        temperature,
                    )

                    dis_action_values.append(action_options[action_index][0])
                    cont_action_values.append([])
                    exps[agent_id].add_instance(
                        state_obs,
                        action_index,
                        q_values.copy(),
                        reward,
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

    def fit(self, epochs: int) -> None:
        temp_states, targets = self.memory.create_targets()

        states = [[torch.tensor(obs).to(self.device) for obs in state] for state in temp_states]

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
                print(f"loss {loss}")  # noqa: T201
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1

        if self.writer is not None:
            self.writer.add_scalar("Loss/Epoch", loss_sum / count, self.curr_epoch)
        self.curr_epoch += 1

    def save_model(self, path: str) -> None:
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model).cpu()),
            (
                # Vis observation
                torch.randn((1, *self.model.visual_input_shape)),
                # Non vis observation
                torch.randn((1, *self.model.nonvis_input_shape)),
            ),
            path,
            opset_version=9,
            input_names=["vis_obs", "nonvis_obs"],
            output_names=["prediction", "action"],
        )

    @classmethod
    def get_state_and_reward(cls, step) -> tuple[tuple[np.ndarray, np.ndarray], np.float32]:  # noqa: ANN001
        state_obs = (
            cls.image_preprocessing(step.obs[0]),
            step.obs[1],
        )

        # step.reward is a number from 0 to 1 representing distance from center of the road

        reward = (  # sigmoid function for reward
            REWARD_MAX / (1 + np.exp((-10 * step.reward) + 4))
            if 0 <= step.reward <= 1
            else step.reward
        )

        return state_obs, reward

    @staticmethod
    def image_preprocessing(img: np.ndarray) -> np.ndarray:
        blurred = (
            gaussian_filter(img, sigma=BLUR_INTENSITY)
            + np.random.normal(0.2, NOISE_INTESITY, img.shape).astype("float32") * NOISE_OPACITY
        )

        # normalizing
        blurred[blurred > 1] = 1
        blurred[blurred < 0] = 0

        slice_starts = (
            blurred.shape[1] - VISUAL_INPUT_SHAPE[1],
            blurred.shape[2] - VISUAL_INPUT_SHAPE[2],
        )
        return blurred[0:, slice_starts[0] :, slice_starts[1] :]
