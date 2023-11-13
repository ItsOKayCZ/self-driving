import numpy as np
import torch
from typing import Tuple
from math import floor
from variables import num_neurons, disc_step_size
from torch.nn import Parameter
import random


class QNetwork(torch.nn.Module):

    def __init__(self, visual_input_shape, nonvis_input_shape, encoding_size, device):
        super(QNetwork, self).__init__()
        height = visual_input_shape[1]
        width = visual_input_shape[2]
        initial_channels = visual_input_shape[0]

        self.output_shape_speed = (1,num_neurons)
        self.output_shape_steer = (1,num_neurons*2)
        self.device = device
        with torch.device(self.device):
            self.visual_input_shape = visual_input_shape
            self.nonvis_input_shape = nonvis_input_shape

            # calculating required size of the dense layer based on the conv layers
            conv_1_hw = self.conv_output_shape((height, width), 5, 1)
            conv_2_hw = self.conv_output_shape(conv_1_hw, 3, 1)
            self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

            # layers
            self.vis_conv1 = torch.nn.Conv2d(initial_channels, 16, 5)
            self.vis_conv2 = torch.nn.Conv2d(16, 32, 3)
            self.nonvis_dense = torch.nn.Linear(nonvis_input_shape[0], 8)
            self.dense = torch.nn.Linear(self.final_flat + 8, encoding_size)
            self.output_speed = torch.nn.Linear(encoding_size, num_neurons)
            self.output_steer = torch.nn.Linear(encoding_size, num_neurons * 2 + 1)

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        nonvis_obs = nonvis_obs.view((-1, self.nonvis_input_shape[0]))

        conv_1 = torch.relu(self.vis_conv1(visual_obs))
        conv_2 = torch.relu(self.vis_conv2(conv_1))
        nonvis_dense = torch.relu(self.nonvis_dense(nonvis_obs))
        # join outputs
        dense = conv_2.reshape([-1, self.final_flat])
        hidden = torch.concat([dense, nonvis_dense], dim=1)

        hidden = self.dense(hidden)
        hidden = torch.relu(hidden)

        output_speed = self.output_speed(hidden)
        output_steer = self.output_steer(hidden)
        return output_speed, output_steer

    def get_actions(self, observation, temperature, use_tensor=False):
        """

        :param observation:
        :param temperature:
        :param use_tensor:
        :return: q_values, actions, action_indices
        """
        if not use_tensor:

            observation = (
                torch.from_numpy(observation[0].reshape(-1,64,64)).to(self.device), torch.from_numpy(observation[1]).to(self.device))

            self.eval()
            with torch.no_grad():
                q_values_speed, q_values_steer = self.forward(observation)
            q_values_speed, q_values_steer = q_values_speed.flatten(1), q_values_steer.flatten(1)
            action_index_speed = self.pick_action(temperature, q_values_speed)
            action_index_steer = self.pick_action(temperature, q_values_steer)
        else:
            self.eval()
            with torch.no_grad():
                observation_input = [
                    observation[0].reshape(-1, 64, 64),
                    observation[1]
                ]
                q_values_speed, q_values_steer = self.forward(observation_input)

            q_values_speed, q_values_steer = q_values_speed.flatten(1), q_values_steer.flatten(1)
            action_index_speed = self.pick_action(temperature, q_values_speed)
            action_index_steer = self.pick_action(temperature, q_values_steer)

        action_speed = action_index_speed[0] * disc_step_size
        action_steer = (action_index_steer[0] - num_neurons) * disc_step_size

        return (q_values_speed, q_values_steer), (action_speed, action_steer), (action_index_speed,action_index_steer)

    def pick_action(self, temperature, q_values):
        if temperature == 0.0:
            action_index = torch.argmax(q_values, dim=1, keepdim=True)
            action_index = action_index.tolist()[0]
        else:
            probs = torch.softmax(q_values / temperature, 1)
            action_index = random.choices(range(len(probs[0])), weights=probs[0])
        return action_index

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
