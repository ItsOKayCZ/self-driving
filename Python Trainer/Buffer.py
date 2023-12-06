import numpy as np
from variables import NUM_NEURONS, DISCOUNT, REWARD_SAME_ACTION
from torch.utils.data import Dataset


class Experience:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.predicted_values = []

    def add_instance(self, observations, action, predicted_values, reward):
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predicted_values.append(predicted_values)

    def flip(self):
        """
        Creates new samples for the dataset by flipping the image and direction of driving
        :return:
        """
        new_observations = [(np.flip(vis, 2), nonvis) for vis, nonvis in self.observations]
        new_actions = [(x[0][0], 2 * NUM_NEURONS - x[1][0]) if x is not None else None for x in self.actions]
        # the new action index for steering is essentialy the action value * -1 expressed with the action index
        new_predicted_values = [(np.flip(x[0], 0), np.flip(x[1], 0)) for x in self.predicted_values]

        new_exp = Experience()
        new_exp.observations = new_observations
        new_exp.actions = new_actions
        new_exp.rewards = self.rewards.copy()
        new_exp.predicted_values = new_predicted_values
        return new_exp

    def calculate_targets(self):
        targets_speed = []
        targets_steer = []
        states = []
        for e, observation in enumerate(self.observations):

            if self.actions[e] is None:
                break

            action_index_speed = self.actions[e][0]
            action_index_steer = self.actions[e][1]
            reward = self.rewards[e]

            if e != 0:
                if self.actions[e][1] == self.actions[e - 1][1]:
                    reward += REWARD_SAME_ACTION

            # we take the matrix of predicted values and for the actions we had taken adjust the value by the reward
            # and the value of the next state
            target_matrix_speed = self.calculate_target(self.predicted_values[e][0],
                                                        self.predicted_values[e + 1][0],
                                                        action_index_speed, reward)
            target_matrix_steer = self.calculate_target(self.predicted_values[e][1],
                                                        self.predicted_values[e + 1][1],
                                                        action_index_steer, reward)
            # adjust
            observation = [arr.astype("float32") for arr in observation]

            states.append(observation)
            targets_speed.append(target_matrix_speed)
            targets_steer.append(target_matrix_steer)

        return states, (targets_speed, targets_steer)

    @staticmethod
    def calculate_target(q_values, next_q_values, action_index, reward):
        target_matrix = q_values.copy()

        # adjust
        value = reward + max(next_q_values[0]) * DISCOUNT
        target_matrix[0, action_index] = value
        target_matrix = target_matrix.astype("float32").reshape(-1)
        return target_matrix


class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add_exp(self, exp):
        if not self.is_full():
            self.buffer.append(exp)

    def is_full(self):
        return len(self.buffer) >= self.size

    def create_targets(self):
        state_dataset = []
        targets_dataset_speed = []
        targets_dataset_steer = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets()
            targets_dataset_speed += targets[0]
            targets_dataset_steer += targets[1]
            state_dataset += states
        return state_dataset, (targets_dataset_speed, targets_dataset_steer)

    def flip_dataset(self):
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

    def wipe(self):
        self.buffer = []


class StateTargetValuesDataset(Dataset):

    def __init__(self, states: list, targets_speed: list, targets_steer: list):
        self.states = states
        self.targets_speed = targets_speed
        self.targets_steer = targets_steer
        if len(states) != len(targets_speed):
            raise ValueError

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], (self.targets_speed[index], self.targets_steer[index])
