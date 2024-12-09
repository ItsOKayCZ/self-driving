MODEL_PATH = "./models"

# QNet parameters
VISUAL_INPUT_SHAPE = (1, 100, 160)
NONVISUAL_INPUT_SHAPE = (2,)

# Hyperparameters
START_TEMPERATURE = 20
REDUCE_TEMPERATURE = 1 / 25
DISCOUNT = 0.95  # devalues future reward
LEARNING_RATE = 0.0005
NUM_TRAINING_EXAMPLES = 50
MAX_TRAINED_EPOCHS = 800

# Reward
REWARD_MAX = 5  # aprox. the maximum reward it will get for staying on the line
STEERING_DISCOUNT = 0.8  # multiplier for steering
SPEED_WEIGHT = 1  # how much of speed to add to reward (speed is from 1 to 0)

# Image preprocessing
BLUR_INTENSITY = 1.1
NOISE_OPACITY = 0.004
NOISE_INTESITY = 2

# Action options
SPEED = 10
SPEED_OPTIONS = [1, 0.5, 0]
STEERING_OPTIONS = [-1, -0.5, 0, 0.5, 1]
ACTION_OPTIONS = tuple((sp, st) for sp in SPEED_OPTIONS for st in STEERING_OPTIONS)
