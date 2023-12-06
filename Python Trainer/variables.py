# QNetwork
ENCODING_SIZE = 258
# Inputs
IMAGE_SHAPE = (1, 64, 64)

# Training
MAX_TRAINED_EPOCHS = 500
NUM_TRAINING_EXAMPLES = 500  # 300
NUM_EVALUATION_EXAMPLES = int(NUM_TRAINING_EXAMPLES / 10)

# Hyperparameters
START_TEMPERATURE = 20
REDUCE_TEMPERATURE = START_TEMPERATURE / 400  # 400 = when we want the to start to be 0
DISCOUNT = 0.75  # devalues future reward
LEARNING_RATE = 0.0005

# Reward
REWARD_SAME_ACTION = 2.0  # will be added to the reward for sticking with the same action

# Discretization
NUM_NEURONS = 2  # number of neurons in the output layer (per variable)
DISC_STEP_SIZE = 1 / (NUM_NEURONS - 1)
