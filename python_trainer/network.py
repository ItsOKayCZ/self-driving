import random
from math import floor
from typing import Tuple

import numpy as np
import torch

# - action shape: [left, forward, right]

action_options = [
    np.array([[1, 0, 1, 0]]),
    np.array([[1, 0, 0, 0]]),
    np.array([[1, 0, 0, 1]]),
]
mirrored_actions = [2, 1, 0]


class QNetwork(torch.nn.Module):
    def __init__(self, visual_input_shape, nonvis_input_shape, device):
        super().__init__()

        self.output_shape = (1, len(action_options))
        self.visual_input_shape = visual_input_shape
        self.nonvis_input_shape = nonvis_input_shape
        self.device = device

        conv_1_args = {"kernels": 16, "kernel_size": 5, "stride": 1}
        conv_2_args = {"kernels": 8, "kernel_size": 3, "stride": 1}
        nonvis_dense_size = 8
        encoding_size = 126

        self.create_layers(conv_1_args, conv_2_args, nonvis_dense_size, encoding_size)

    def create_layers(
        self,
        conv_1_args: dict[str, int],
        conv_2_args: dict[str, int],
        nonvis_dense_size: int,
        encoding_size: int,
    ) -> None:
        height = self.visual_input_shape[1]
        width = self.visual_input_shape[2]
        initial_channels = self.visual_input_shape[0]

        # calculating required size of the dense layer based on the conv layers
        conv_1_hw = self.conv_output_shape(
            (height, width),
            conv_1_args["kernel_size"],
            conv_1_args["stride"],
        )
        conv_2_hw = self.conv_output_shape(
            conv_1_hw,
            conv_2_args["kernel_size"],
            conv_2_args["stride"],
        )

        self.final_flat_size = conv_2_hw[0] * conv_2_hw[1] * conv_2_args["kernels"]

        with torch.device(self.device):
            # layers
            self.conv_1 = torch.nn.Conv2d(
                initial_channels,
                conv_1_args["kernels"],
                conv_1_args["kernel_size"],
                stride=conv_1_args["stride"],
            )

            self.conv_2 = torch.nn.Conv2d(
                conv_1_args["kernels"],
                conv_2_args["kernels"],
                conv_2_args["kernel_size"],
                stride=conv_2_args["stride"],
            )
            self.nonvis_dense = torch.nn.Linear(self.nonvis_input_shape[0], nonvis_dense_size)
            self.dense_1 = torch.nn.Linear(self.final_flat_size + nonvis_dense_size, encoding_size)
            self.dense_2 = torch.nn.Linear(encoding_size, self.output_shape[1])

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        nonvis_obs = nonvis_obs.view((-1, self.nonvis_input_shape[0]))

        conv_1 = torch.relu(self.conv_1(visual_obs))
        conv_2 = torch.relu(self.conv_2(conv_1))
        nonvis_dense = torch.relu(self.nonvis_dense(nonvis_obs))
        hidden = conv_2.reshape([-1, self.final_flat_size])
        hidden = torch.concat([hidden, nonvis_dense], dim=1)
        hidden = self.dense_1(hidden)
        hidden = torch.relu(hidden)
        output = self.dense_2(hidden)
        return output

    def get_actions(self, observation, temperature, use_tensor=False):
        """
        Get the q values, if positive we do the action
        :param observation:
        :return q_values:
        """

        if not use_tensor:
            observation = (
                torch.from_numpy(observation[0]).to(self.device),
                torch.from_numpy(observation[1]).to(self.device),
            )
            self.eval()
            with torch.no_grad():
                q_values = self.forward(observation)
            q_values = q_values.flatten(1)
            if temperature == 0:
                action_index = torch.argmax(q_values, dim=1, keepdim=True)
            else:
                probs = torch.softmax(q_values / temperature, 1)
                action_index = random.choices(range(len(action_options)), weights=probs[0])
            q_values = q_values.cpu().detach().numpy().flatten()

        else:
            self.eval()
            with torch.no_grad():
                q_values = self.forward(observation)
            q_values = q_values.view((-1, self.output_shape[1]))
            if temperature == 0:
                action_index = torch.argmax(q_values, dim=1, keepdim=True)
            else:
                probs = torch.softmax(q_values / temperature, 1)
                action_index = random.choices(range(len(action_options)), weights=probs)

        return q_values, action_index[0]

    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        return h, w
