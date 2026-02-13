import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from config import (
    GRID_SIZE, GOAL_POS, GOAL_RADIUS, STEP_SIZE,
    MAX_STEPS_PER_EPISODE, REF_POINTS, AGENT_NAMES,
)


class MultiAgentGridEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_agent_grid_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = list(AGENT_NAMES)
        self.render_mode = render_mode

        self._goal_map = {
            name: np.array(GOAL_POS[i], dtype=np.float32)
            for i, name in enumerate(self.possible_agents)
        }
        self._ref_points = np.array(REF_POINTS, dtype=np.float32)

        self._positions = {}
        self._done_flags = {}
        self._steps = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        max_dist = np.sqrt(GRID_SIZE**2 + (GRID_SIZE / 2) ** 2)
        return spaces.Box(
            low=0.0, high=max_dist, shape=(4,), dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        rng = np.random.default_rng(seed)

        self._positions = {
            name: rng.uniform(0.0, GRID_SIZE, size=(2,)).astype(np.float32)
            for name in self.agents
        }
        self._done_flags = {name: False for name in self.agents}
        self._steps = 0

        observations = {name: self._get_obs(name) for name in self.agents}
        infos = {name: {} for name in self.agents}
        return observations, infos

    def step(self, actions):
        current_agents = list(self.agents)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        self._steps += 1

        for agent in current_agents:
            action = np.asarray(actions[agent], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            norm = np.linalg.norm(action)
            if norm > 1e-8:
                direction = action / norm
            else:
                direction = np.zeros(2, dtype=np.float32)

            new_pos = self._positions[agent] + direction * STEP_SIZE
            new_pos = np.clip(new_pos, 0.0, GRID_SIZE)
            self._positions[agent] = new_pos

            obs = self._get_obs(agent)
            observations[agent] = obs

            goal = self._goal_map[agent]
            dist_to_goal = float(np.linalg.norm(self._positions[agent] - goal))
            terminated = dist_to_goal <= GOAL_RADIUS

            if terminated:
                reward = 10.0
            else:
                distance_penalty = -np.sum(np.abs(obs - 1.0))
                reward = distance_penalty - 0.01

            truncated = self._steps >= MAX_STEPS_PER_EPISODE

            terminations[agent] = terminated
            truncations[agent] = truncated
            rewards[agent] = float(reward)
            infos[agent] = {"dist_to_goal": dist_to_goal}

            if terminated or truncated:
                self._done_flags[agent] = True

        self.agents = [a for a in current_agents if not self._done_flags[a]]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        pos = self._positions[agent]
        diffs = self._ref_points - pos
        dists = np.linalg.norm(diffs, axis=1)
        return dists.astype(np.float32)

    def render(self):
        print(f"Step {self._steps:3d}")
        for name in self.possible_agents:
            pos = self._positions.get(name)
            if pos is not None:
                goal = self._goal_map[name]
                dist = np.linalg.norm(pos - goal)
                done = self._done_flags.get(name, False)
                status = "DONE" if done else f"d={dist:.3f}"
                print(
                    f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f}) -> "
                    f"goal ({goal[0]:.1f}, {goal[1]:.1f}) [{status}]"
                )

    def close(self):
        pass
