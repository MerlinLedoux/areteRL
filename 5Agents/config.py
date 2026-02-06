GRID_SIZE = 2.0

EDGE_AGENTS = {
    "bottom": {"fixed_axis": "y", "fixed_value": 0.0},
    "left":   {"fixed_axis": "x", "fixed_value": 0.0},
    "right":  {"fixed_axis": "x", "fixed_value": 2.0},
    "top":    {"fixed_axis": "y", "fixed_value": 2.0},
}

GOAL_POS_CENTRAL = (1.0, 1.0)
GOAL_RADIUS_CENTRAL = 0.1
GOAL_RADIUS_EDGE = 0.1

STEP_SIZE = 0.1
MAX_STEPS_PER_EPISODE = 100

REWARD_GOAL = 10.0
TIME_PENALTY = -0.01

# PPO hyperparameters — edge agents
PPO_EDGE_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# PPO hyperparameters — central agent
PPO_CENTRAL_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

TIMESTEPS_PER_ROUND = 20_000
NUM_TRAINING_ROUNDS = 10

MODEL_PATH_EDGE = "models/ppo_edge"
MODEL_PATH_CENTRAL = "models/ppo_central"

N_EVAL_EPISODES = 50
