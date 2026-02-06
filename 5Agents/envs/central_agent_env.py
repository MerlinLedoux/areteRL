import math
import numpy as np
import gymnasium
from gymnasium import spaces

from config import (
    GRID_SIZE, MAX_STEPS_PER_EPISODE,
    GOAL_POS_CENTRAL, GOAL_RADIUS_CENTRAL, REWARD_GOAL, TIME_PENALTY,
)


class CentralAgentEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, world, render_mode=None):
        super().__init__()
        self.world = world
        self.render_mode = render_mode
        self.goal = np.array(GOAL_POS_CENTRAL, dtype=np.float32)
        self.steps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        max_dist = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
        self.observation_space = spaces.Box(
            low=0.0, high=max_dist, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(seed=seed)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.world.apply_central_action(action)
        self.world.step_other_agents(exclude_agent="central")
        self.steps += 1

        obs = self._get_obs()
        desired_dist = 1.0
        distance_penalty = -np.sum(np.abs(obs - desired_dist))

        terminated = self.world.all_goals_reached()

        if terminated:
            reward = REWARD_GOAL
        else:
            reward = TIME_PENALTY + distance_penalty

        truncated = self.steps >= MAX_STEPS_PER_EPISODE
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return self.world.get_central_obs()
