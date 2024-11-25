import torch
from network import QNetwork
from variables import ACTION_OPTIONS

# NOTE: For use in unity, different input and output shape is needed
# Inputs:
# - shape: (-1, 120, 160, 1) -> Vis observation
# - shape: (-1, 1, 1, 3) -> Nonvis observation
# - shape: (-1, 1, 1, 4) -> Action mask
#
# Outputs:
# - version_number: shape (1, 1, 1, 1) = [3]
# - memory_size: shape (1, 1, 1, 1) = [0]
# - discrete_actions: shape (1, 1, 1, 4) = [[2, 2, 2, 2]]
# - discrete_action_output_shape: shape (1, 1, 1, 4) -> network.action_options
# - deterministic_discrete_actions: shape (1, 1, 1, 4) -> network.action_options


class WrapperNet(torch.nn.Module):
    def __init__(self, qnet: QNetwork) -> None:
        super().__init__()
        self.qnet = qnet

    def forward(self, vis_obs: torch.Tensor, nonvis_obs: torch.Tensor) -> tuple[torch.Tensor, int]:
        qnet_result, action_index = self.qnet.get_actions_tensor(
            (vis_obs, nonvis_obs),
            temperature=0,
        )
        action_options = ACTION_OPTIONS
        output = action_options[action_index]

        return qnet_result, output
