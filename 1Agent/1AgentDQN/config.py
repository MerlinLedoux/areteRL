GRID_SIZE = 2.0

GOAL_POS = (1, 1)
GOAL_RADIUS = 0.1

STEP_SIZE = 0.1

MAX_STEPS_PER_EPISODE = 100

# DQN hyperparameters
DQN_CONFIG = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "target_update_interval": 500,
    "train_freq": 4,
    "learning_starts": 1000,
}

TOTAL_TIMESTEPS = 100_000

MODEL_PATH = "models/dqn_grid"

N_EVAL_EPISODES = 50
