import numpy as np
from network import mirrored_actions
from torch.utils.data import Dataset
from variables import DISCOUNT, STEERING_DISCOUNT


class Experience:
    def __init__(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.predicted_values = []

    def add_instance(
        self,
        observations: list[np.ndarray],
        action: int,
        predicted_values: np.ndarray,
        reward: np.float32,
    ) -> None:
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predicted_values.append(predicted_values)

    def flip(self) -> "Experience":
        new_observations = [(np.flip(vis, 2), nonvis) for vis, nonvis in self.observations]

        new_actions = [mirrored_actions[x] if x is not None else None for x in self.actions]

        new_predicted_values = [np.flip(x, 0) for x in self.predicted_values]

        new_exp = Experience()
        new_exp.observations = new_observations
        new_exp.actions = new_actions
        new_exp.rewards = self.rewards.copy()
        new_exp.predicted_values = new_predicted_values
        return new_exp

    def calculate_targets(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        targets = []
        states = []
        for e, observation in enumerate(self.observations):
            if self.actions[e] is None:
                break

            action_index = self.actions[e]
            reward = self.rewards[e]

            if action_index != 1:  # is steering
                reward *= STEERING_DISCOUNT

            # we take the matrix of predicted values and for the actions
            # we had taken adjust the value by the reward and the value of the next state
            target_matrix = self.predicted_values[e].copy()

            # adjust
            target_matrix[action_index] = reward + max(self.predicted_values[e + 1]) * DISCOUNT
            states.append([arr.astype("float32") for arr in observation])
            targets.append(target_matrix.astype("float32"))

        return states, targets

    def __len__(self) -> int:
        return len(self.observations)


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.buffer: list[Experience] = []

    def add_exp(self, exp: Experience) -> None:
        if not self.is_full():
            self.buffer.append(exp)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size

    def __len__(self) -> int:
        return sum([len(x) for x in self.buffer])

    def size(self) -> int:
        return len(self.buffer)

    def create_targets(self) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        state_dataset = []
        targets_dataset = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets()
            targets_dataset += targets
            state_dataset += states
        return state_dataset, targets_dataset

    def flip_dataset(self) -> None:
        """
        Mirrors the image and action data in dataset, effectively doubles it.
        :return:
        """
        new_exps = []
        for exp in self.buffer:
            new_exp = exp.flip()
            new_exps.append(new_exp)

        for new_exp in new_exps:
            self.buffer.append(new_exp)

    def wipe(self) -> None:
        self.buffer = []


class StateTargetValuesDataset(Dataset):
    def __init__(self, states: list, targets: list) -> None:
        self.states = states
        self.targets = targets
        if len(states) != len(targets):
            raise ValueError

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.states[index], self.targets[index]
