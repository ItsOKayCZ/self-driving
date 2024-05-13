import numpy as np
import torch
from typing import Tuple
from math import floor
from torch.nn import Parameter
import random

# - action shape: [left, forward, right]

action_options = [np.array([[1, 0, 1, 0]]), np.array([[1, 0, 0, 0]]), np.array([[1, 0, 0, 1]])]
mirrored_actions = [2, 1, 0]


class QNetwork(torch.nn.Module):

    def __init__(self, visual_input_shape, nonvis_input_shape, encoding_size, device):
        super(QNetwork, self).__init__()
        height = visual_input_shape[1]
        width = visual_input_shape[2]
        initial_channels = visual_input_shape[0]

        self.device = device

        with torch.device(self.device):
            self.output_shape = (1, len(action_options))
            self.visual_input_shape = visual_input_shape
            self.nonvis_input_shape = nonvis_input_shape
            # calculating required size of the dense layer based on the conv layers
            conv1_hw = self.conv_output_shape((height, width), 5, 1)
            maxpool1_hw = self.conv_output_shape(conv1_hw, 2, 2)
            conv2_hw = self.conv_output_shape(maxpool1_hw, 3, 1)
            maxpool2_hw = self.conv_output_shape(conv2_hw, 2, 2)
            self.final_flat = maxpool2_hw[0] * maxpool2_hw[1] * 32
            print(self.final_flat)

            # layers
            self.conv1 = torch.nn.Conv2d(initial_channels, 16, 5)
            self.maxpool1 = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(16, 32, 3)
            self.maxpool2 = torch.nn.MaxPool2d(2, 2)
            self.nonvis_dense = torch.nn.Linear(nonvis_input_shape[0], 8)
            self.dense1 = torch.nn.Linear(self.final_flat + 8, encoding_size)
            self.dense2 = torch.nn.Linear(encoding_size, self.output_shape[1])

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        nonvis_obs = nonvis_obs.view((-1, self.nonvis_input_shape[0]))

        conv_1 = torch.relu(self.conv1(visual_obs))
        maxpool1 = self.maxpool1(conv_1)
        conv_2 = torch.relu(self.conv2(maxpool1))
        maxpool2 = self.maxpool2(conv_2)

        nonvis_dense = torch.relu(self.nonvis_dense(nonvis_obs))

        hidden = maxpool2.reshape([-1, self.final_flat])
        hidden = torch.concat([hidden, nonvis_dense], dim=1)
        hidden = self.dense1(hidden)
        hidden = torch.relu(hidden)
        output = self.dense2(hidden)
        return output

    def get_actions(self, observation, temperature, use_tensor=False):
        """
        Get the q values, if positive we do the action
        :param observation:
        :return q_values:
        """

        if not use_tensor:
            observation = (
                torch.from_numpy(observation[0]).to(self.device), torch.from_numpy(observation[1]).to(self.device))
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
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w
