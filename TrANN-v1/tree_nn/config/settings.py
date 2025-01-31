"""Configuration settings for the Tree Neural Network models."""

# Model parameters
MODEL_PARAMS = {
    "hidden_dim": 20,
    "output_dim": 1,
    "num_trees": 20,
    "epochs": 10000,
    "learning_rate": 0.01,
    "random_state": 42,
}

# Data parameters
TRAIN_TEST_SPLIT = 0.2
DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
