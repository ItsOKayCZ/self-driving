
max_trained_epochs = 500

# Hyperparameters
exploration_chance_start = 0.90
exploration_reduce = 0.98 # 0.75
num_training_examples = 20
discount = 0.95 # devalues future reward
learning_rate = 0.0005


# Reward
reward_same_action = 2.0 # will be added to the reward for sticking with the same action
