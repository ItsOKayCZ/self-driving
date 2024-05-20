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
            hw = self.conv_output_shape((height, width), 8, 4, 4) # conv1
            hw = self.conv_output_shape(hw, 2, 2) # maxpool
            hw = self.conv_output_shape(hw, 4, 2) # conv 2
            hw = self.conv_output_shape(hw, 3, 1) # conv 2
            # hw = self.conv_output_shape(hw, 2, 2) # max pool
            self.final_flat = hw[0] * hw[1] * 64
            print(self.final_flat)

            # layers
            self.dense1 = torch.nn.Linear(self.final_flat + 8, encoding_size)
            self.dense2 = torch.nn.Linear(encoding_size, self.output_shape[1])

            self.conv1 = torch.nn.Conv2d(initial_channels, 32, 8 ,stride=4,padding=4)
            self.conv2 = torch.nn.Conv2d(32, 64, 4,stride=2)
            self.conv3 = torch.nn.Conv2d(64, 64, 3)

            self.maxpool1 = torch.nn.MaxPool2d(2, 2)
            self.maxpool2 = torch.nn.MaxPool2d(2, 2)

            self.nonvis_dense = torch.nn.Linear(nonvis_input_shape[0], 8)

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        nonvis_obs = nonvis_obs.view((-1, self.nonvis_input_shape[0]))

        visual_y = torch.relu(self.conv1(visual_obs))
        visual_y = self.maxpool1(visual_y)
        visual_y = torch.relu(self.conv2(visual_y))
        visual_y = torch.relu(self.conv3(visual_y))
        visual_y = visual_y.reshape([-1, self.final_flat])
        # maxpool2 = self.maxpool2(conv_2)

        nonvis_y = torch.relu(self.nonvis_dense(nonvis_obs))

        comb_y = torch.concat([visual_y, nonvis_y], dim=1)
        comb_y = self.dense1(comb_y)
        comb_y = torch.relu(comb_y)
        output = self.dense2(comb_y)
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
