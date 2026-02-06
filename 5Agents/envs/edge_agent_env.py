import math
import numpy as np
import gymnasium
from gymnasium import spaces

from config import GRID_SIZE, MAX_STEPS_PER_EPISODE, GOAL_RADIUS_EDGE, REWARD_GOAL, TIME_PENALTY


class EdgeAgentEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, world, agent_name, render_mode=None):
        super().__init__()
        self.world = world
        self.agent_name = agent_name
        self.render_mode = render_mode
        self.steps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        max_dist = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
        self.observation_space = spaces.Box(
            low=0.0, high=max_dist, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(seed=seed)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.world.apply_edge_action(self.agent_name, action[0])
        self.world.step_other_agents(exclude_agent=self.agent_name)
        self.steps += 1

        obs = self._get_obs()
        pos = self.world.edge_positions[self.agent_name]
        dist_to_mid = abs(pos - 1.0)

        terminated = self.world.all_goals_reached()

        if terminated:
            reward = REWARD_GOAL
        else:
            reward = -dist_to_mid + TIME_PENALTY

        truncated = self.steps >= MAX_STEPS_PER_EPISODE
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return self.world.get_edge_obs(self.agent_name)
