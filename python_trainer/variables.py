MODEL_PATH = "./models"

# QNet parameters
VISUAL_INPUT_SHAPE = (1, 85, 160)
NONVISUAL_INPUT_SHAPE = (1,)

# Hyperparameters
START_TEMPERATURE = 10
REDUCE_TEMPERATURE = 1 / 25
DISCOUNT = 0.95  # devalues future reward
LEARNING_RATE = 0.0005
NUM_TRAINING_EXAMPLES = 7500
MAX_TRAINED_EPOCHS = 1000

# Reward
REWARD_MAX = 10  # aprox. the maximum reward it will get for staying on the line
STEERING_DISCOUNT = 0.8  # multiplier for steering


# Image preprocessing
BLUR_INTENSITY = 1.2
NOISE_OPACITY = 0.004
NOISE_INTESITY = 8
