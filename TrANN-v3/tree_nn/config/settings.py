"""Configuration settings for the Tree Neural Network models."""

# Model parameters
MODEL_PARAMS = {
    'epochs': 1000,
    'hidden_dim': 64,
    'learning_rate': 0.01,
    'num_trees': 20,
    'batch_size': 32,
    'random_state': 42,
    'early_stopping_patience': 50,
    'l2_strength': 0.001,          # L2 regularization strength
    'dropout_rate': 0.2,           # Dropout rate for neural network layers
    'weight_decay': 0.0005,        # Weight decay for optimizer
    'regularize_trees': True       # Whether to apply regularization to tree parameters
}

# Data parameters
TRAIN_TEST_SPLIT = 0.2
DATA_URL = 'http://lib.stat.cmu.edu/datasets/boston' 