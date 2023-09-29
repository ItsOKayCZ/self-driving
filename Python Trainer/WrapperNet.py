from network import QNetwork
from typing import List
import torch
from torch.nn import Parameter
from torch.nn.functional import one_hot

class WrapperNet(torch.nn.Module):
    def __init__(self, qnet: QNetwork):
        super(WrapperNet, self).__init__()
        self.qnet = qnet

    def forward(self, vis_obs: torch.Tensor, nonvis_obs: torch.Tensor):
        qnet_result, action_index = self.qnet.get_actions((vis_obs, nonvis_obs),temperature = 0, use_tensor=True)
        action_options = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1]
        ])
        output = action_options[action_index]

        return qnet_result, output