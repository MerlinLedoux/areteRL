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

        self._positions = {}
        self._done_flags = {}
        self._steps = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        max_dist = np.sqrt(GRID_SIZE**2 + (GRID_SIZE / 2) ** 2)
        return spaces.Box(
            low=-1, high=max_dist, shape=(4,), dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        rng = np.random.default_rng(seed)

        self._positions = {}
        for name in self.agents : 
            if name == "agent_1":
                y = rng.uniform(0.0, GRID_SIZE)
                self._positions[name] = np.array([2.0, y], dtype=np.float32)
            elif name == "agent_2":
                x = rng.uniform(0.0, GRID_SIZE)
                self._positions[name] = np.array([x, 0.0], dtype=np.float32)
            elif name == "agent_3":
                y = rng.uniform(0.0, GRID_SIZE)
                self._positions[name] = np.array([0.0, y], dtype=np.float32)
            elif name == "agent_4":
                x = rng.uniform(0.0, GRID_SIZE)
                self._positions[name] = np.array([x, 2.0], dtype=np.float32)
            else:
                self._positions[name] = rng.uniform(
                    0.0, GRID_SIZE, size=(2,)
                ).astype(np.float32)
        
        
        self._done_flags = {name: False for name in self.agents}
        self._steps = 0

        observations = {name: self._get_obs(name) for name in self.agents}
        infos = {name: {} for name in self.agents}
        return observations, infos

    def step(self, actions):
        current_agents = list(self.agents)
        self._steps += 1

        # === Phase 1 : Deplacer TOUS les agents (avant tout calcul d'obs) ===
        forbidden_penalties = {}
        for agent in current_agents:
            action = np.asarray(actions[agent], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Penalite mouvement interdit (sur l'action brute)
            forbidden_penalty = 0.0
            if agent in ("agent_1", "agent_3"):
                forbidden_penalty = -abs(float(action[0])) * 0.1
            elif agent in ("agent_2", "agent_4"):
                forbidden_penalty = -abs(float(action[1])) * 0.1
            forbidden_penalties[agent] = forbidden_penalty

            # Mouvement
            norm = np.linalg.norm(action)
            if norm > 1e-8:
                direction = action / norm
            else:
                direction = np.zeros(2, dtype=np.float32)

            if agent in ("agent_1", "agent_3"):
                direction[0] = 0.0
            if agent in ("agent_2", "agent_4"):
                direction[1] = 0.0

            new_pos = self._positions[agent] + direction * STEP_SIZE
            self._positions[agent] = np.clip(new_pos, 0.0, GRID_SIZE)

        # === Phase 2 : Calculer obs et rewards (toutes les positions a jour) ===
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        newly_terminated = []

        for agent in current_agents:
            obs = self._get_obs(agent)
            observations[agent] = obs

            # Penalite basee sur l'observation : guider chaque distance vers 1.0
            valid_dists = obs[obs >= 0.0]
            mesh_penalty = -float(np.sum(np.abs(valid_dists - 1.0))) * 0.3

            time_penalty = -0.01

            # Terminaison
            goal = self._goal_map[agent]
            dist_to_goal = float(np.linalg.norm(self._positions[agent] - goal))
            terminated = dist_to_goal <= GOAL_RADIUS
            truncated = self._steps >= MAX_STEPS_PER_EPISODE

            # Reward
            reward = forbidden_penalties[agent] + mesh_penalty + time_penalty
            if terminated:
                reward += 5.0
                newly_terminated.append(agent)

            terminations[agent] = terminated
            truncations[agent] = truncated
            rewards[agent] = float(reward)
            infos[agent] = {"dist_to_goal": dist_to_goal}

            if terminated or truncated:
                self._done_flags[agent] = True

        # Grande recompense quand TOUS les agents ont fini
        if all(self._done_flags.values()) and newly_terminated:
            for agent in current_agents:
                rewards[agent] += 20.0

        self.agents = [a for a in current_agents if not self._done_flags[a]]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        pos = self._positions[agent]
        pos_a0 = self._positions["agent_0"]

        def _dist(target):
            return float(np.linalg.norm(pos - np.asarray(target, dtype=np.float32)))

        if agent == "agent_1":
            obs = [-1.0, _dist([2, 0]), _dist(pos_a0), _dist([2, 2])]
        elif agent == "agent_2":
            obs = [_dist([2, 0]), -1.0, _dist([0, 0]), _dist(pos_a0)]
        elif agent == "agent_3":
            obs = [_dist(pos_a0), _dist([0, 0]), -1.0, _dist([0, 2])]
        elif agent == "agent_4":
            obs = [_dist([2, 2]), _dist(pos_a0), _dist([0, 2]), -1.0]
        else:
            obs = [
                _dist(self._positions["agent_1"]),
                _dist(self._positions["agent_2"]),
                _dist(self._positions["agent_3"]),
                _dist(self._positions["agent_4"]),
            ]

        return np.array(obs, dtype=np.float32)


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
