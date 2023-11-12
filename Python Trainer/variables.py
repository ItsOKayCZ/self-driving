
max_trained_epochs = 500

# Hyperparameters
start_temperature = 20
reduce_temperature = 1/10 # 0.75
num_training_examples = 100
discount = 0.95 # devalues future reward
learning_rate = 0.0005


# Reward
reward_same_action = 2.0 # will be added to the reward for sticking with the same action

# Discretization
num_neurons = 10 # number of neurons in the output layer (per variable)
disc_step_size = 1/num_neurons
