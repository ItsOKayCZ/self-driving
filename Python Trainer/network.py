import numpy as np
import torch
from typing import Tuple
from math import floor
from torch.nn import Parameter


# - action shape: [forward, backward, right, left]

action_options = [np.array([[1, 0, 1, 0]]), np.array([[1, 0, 0, 0]]), np.array([[1, 0, 0, 1]])]
mirrored_actions = [2,1,0]

class QNetwork(torch.nn.Module):

    def __init__(self, visual_input_shape, nonvis_input_shape, encoding_size):
        super(QNetwork, self).__init__()
        height = visual_input_shape[1]
        width = visual_input_shape[2]
        initial_channels = visual_input_shape[0]

        self.output_shape = (1,len(action_options))
        self.visual_input_shape = visual_input_shape
        self.nonvis_input_shape = nonvis_input_shape
        # calculating required size of the dense layer based on the conv layers
        conv_1_hw = self.conv_output_shape((height, width), 5, 1)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 3, 1)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        # layers
        self.conv1 = torch.nn.Conv2d(initial_channels, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.nonvis_dense = torch.nn.Linear(nonvis_input_shape[0], 8)
        self.dense1 = torch.nn.Linear(self.final_flat + 8, encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size, self.output_shape[1])

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        nonvis_obs = nonvis_obs.view((-1, self.nonvis_input_shape[0]))

        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        nonvis_dense = torch.relu(self.nonvis_dense(nonvis_obs))
        hidden = conv_2.reshape([-1, self.final_flat])
        hidden = torch.concat([hidden, nonvis_dense], dim=1)
        hidden = self.dense1(hidden)
        hidden = torch.relu(hidden)
        output = self.dense2(hidden)
        return output

    def get_actions(self, observation, use_tensor=False):
        """
        Get the q values, if positive we do the action
        :param observation:
        :return q_values:
        """
        
        if not use_tensor:
            observation = (torch.from_numpy(observation[0]), torch.from_numpy(observation[1]))
            self.eval()
            with torch.no_grad():
                q_values = self.forward(observation)
            q_values = q_values.numpy().flatten()
            action_index = np.argmax(q_values)

        else:
            self.eval()
            with torch.no_grad():
                q_values = self.forward(observation)
            q_values = q_values.flatten().view((-1, self.output_shape[0]))
            action_index = torch.argmax(q_values, dim=1, keepdim=True)

        return q_values, action_index

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
