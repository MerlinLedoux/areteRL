# --------------- Environment ------------------------------------------------
GRID_SIZE = 2.0

GOAL_POS = ((1, 1), (2, 1), (1, 0), (0, 1), (1, 2))
GOAL_RADIUS = 0.1

STEP_SIZE = 0.1

MAX_STEPS_PER_EPISODE = 100

REF_POINTS = ((1, 0), (0, 1), (2, 1), (1, 2))

N_AGENTS = 5
AGENT_NAMES = [f"agent_{i}" for i in range(N_AGENTS)]

# --------------- PPO --------------------------------------------------------
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

HIDDEN_SIZES = (64, 64)

TOTAL_TIMESTEPS = 200_000

MODEL_DIR = "models"
MODEL_PATH = "models/ppo_multi.pt"

SAVE_INTERVAL = 10_000

# --------------- Evaluation --------------------------------------------------
N_EVAL_EPISODES = 50
