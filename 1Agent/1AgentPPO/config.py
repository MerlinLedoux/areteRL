GRID_SIZE = 2.0

GOAL_POS = (1, 1)
GOAL_RADIUS = 0.1

STEP_SIZE = 0.1

MAX_STEPS_PER_EPISODE = 100

# PPO hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

TOTAL_TIMESTEPS = 100_000

MODEL_PATH = "models/ppo_grid"

N_EVAL_EPISODES = 50
