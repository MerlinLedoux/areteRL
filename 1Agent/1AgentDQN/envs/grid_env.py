import numpy as np
import math
import gymnasium
from gymnasium import spaces

from config import GRID_SIZE, GOAL_POS, GOAL_RADIUS, STEP_SIZE, MAX_STEPS_PER_EPISODE

# Actions: 0=haut, 1=bas, 2=gauche, 3=droite
ACTIONS_DELTA = {
    0: np.array([0.0, STEP_SIZE]),
    1: np.array([0.0, -STEP_SIZE]),
    2: np.array([-STEP_SIZE, 0.0]),
    3: np.array([STEP_SIZE, 0.0]),
}


class GridWorldEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.size = GRID_SIZE
        self.goal = np.array(GOAL_POS, dtype=np.float32)
        self.goal_radius = GOAL_RADIUS

        self.action_space = spaces.Discrete(4)
        max_dist = math.sqrt(self.size**2 + (self.size / 2)**2)
        self.observation_space = spaces.Box(
            low=0.0, high=max_dist, shape=(4,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.agent_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.np_random.uniform(
            low=0.0, high=self.size, size=(2,)
        ).astype(np.float32)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        delta = ACTIONS_DELTA[int(action)]
        new_pos = self.agent_pos + delta

        if 0.0 <= new_pos[0] <= self.size and 0.0 <= new_pos[1] <= self.size:
            self.agent_pos = new_pos.astype(np.float32)

        self.steps += 1

        obs = self._get_obs()
        desired_dist = 1.0
        distance_penalty = -np.sum(np.abs(obs - desired_dist))

        time_penalty = -0.01

        dist = np.linalg.norm(self.agent_pos - self.goal)
        terminated = dist <= self.goal_radius

        if terminated:
            reward = 10.0
        else:
            reward = time_penalty + distance_penalty

        truncated = self.steps >= MAX_STEPS_PER_EPISODE

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.array([
            math.sqrt((1 - self.agent_pos[0])**2 + (0 - self.agent_pos[1])**2),
            math.sqrt((0 - self.agent_pos[0])**2 + (1 - self.agent_pos[1])**2),
            math.sqrt((2 - self.agent_pos[0])**2 + (1 - self.agent_pos[1])**2),
            math.sqrt((1 - self.agent_pos[0])**2 + (2 - self.agent_pos[1])**2),
        ], dtype=np.float32)

        return obs

    def render(self):
        dist = np.linalg.norm(self.agent_pos - self.goal)
        print(
            f"Step {self.steps:3d} | "
            f"Agent ({self.agent_pos[0]:.2f}, {self.agent_pos[1]:.2f}) | "
            f"Goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) | "
            f"Dist {dist:.3f}"
        )

    def close(self):
        pass
