MODEL_PATH = "./models"

# QNet parameters
VISUAL_INPUT_SHAPE = (1, 120, 160)
NONVISUAL_INPUT_SHAPE = (1,)
ENCODING_SIZE = 126

# Hyperparameters
START_TEMPERATURE = 20
REDUCE_TEMPERATURE = 1 / 15  # 0.75
DISCOUNT = 0.95  # devalues future reward
LEARNING_RATE = 0.0005
NUM_TRAINING_EXAMPLES = 2000
MAX_TRAINED_EPOCHS = 500

# Reward
REWARD_SAME_ACTION = 2.0  # will be added to the reward for sticking with the same action
