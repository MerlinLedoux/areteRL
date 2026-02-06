import numpy as np

from config import GRID_SIZE, EDGE_AGENTS, STEP_SIZE, GOAL_RADIUS_EDGE, GOAL_RADIUS_CENTRAL, GOAL_POS_CENTRAL


class MultiAgentWorld:
    """Etat partage du monde multi-agent. Pas un gymnasium.Env."""

    def __init__(self):
        self.central_pos = None
        self.edge_positions = {}
        self.step_count = 0

        self.edge_policy = None
        self.central_policy = None

        self._rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.central_pos = self._rng.uniform(0.0, GRID_SIZE, size=(2,)).astype(np.float32)

        for name in EDGE_AGENTS:
            self.edge_positions[name] = float(self._rng.uniform(0.0, GRID_SIZE))

        self.step_count = 0

    def get_edge_2d_pos(self, agent_name):
        cfg = EDGE_AGENTS[agent_name]
        free_val = self.edge_positions[agent_name]
        if cfg["fixed_axis"] == "y":
            return np.array([free_val, cfg["fixed_value"]], dtype=np.float32)
        else:
            return np.array([cfg["fixed_value"], free_val], dtype=np.float32)

    def get_edge_obs(self, agent_name):
        edge_pos = self.get_edge_2d_pos(agent_name)
        dist = np.linalg.norm(edge_pos - self.central_pos)
        return np.array([dist], dtype=np.float32)

    def is_edge_at_goal(self, agent_name):
        return abs(self.edge_positions[agent_name] - 1.0) <= GOAL_RADIUS_EDGE

    def is_central_at_goal(self):
        goal = np.array(GOAL_POS_CENTRAL, dtype=np.float32)
        return np.linalg.norm(self.central_pos - goal) <= GOAL_RADIUS_CENTRAL

    def all_goals_reached(self):
        if not self.is_central_at_goal():
            return False
        for name in EDGE_AGENTS:
            if not self.is_edge_at_goal(name):
                return False
        return True

    def get_central_obs(self):
        dists = []
        for name in EDGE_AGENTS:
            edge_pos = self.get_edge_2d_pos(name)
            dists.append(np.linalg.norm(self.central_pos - edge_pos))
        return np.array(dists, dtype=np.float32)

    def apply_edge_action(self, agent_name, action_1d):
        action_1d = float(np.clip(action_1d, -1.0, 1.0))
        if abs(action_1d) > 1e-8:
            direction = action_1d / abs(action_1d)
        else:
            direction = 0.0
        new_val = self.edge_positions[agent_name] + direction * STEP_SIZE
        if 0.0 <= new_val <= GRID_SIZE:
            self.edge_positions[agent_name] = new_val

    def apply_central_action(self, action_2d):
        action_2d = np.clip(action_2d, -1.0, 1.0)
        norm = np.linalg.norm(action_2d)
        if norm > 1e-8:
            direction = action_2d / norm
        else:
            direction = np.zeros(2, dtype=np.float32)
        new_pos = self.central_pos + direction * STEP_SIZE
        if 0.0 <= new_pos[0] <= GRID_SIZE and 0.0 <= new_pos[1] <= GRID_SIZE:
            self.central_pos = new_pos.astype(np.float32)

    def step_other_agents(self, exclude_agent):
        # Batch edge agents
        edge_names = [n for n in EDGE_AGENTS if n != exclude_agent]
        if edge_names and self.edge_policy is not None:
            obs_batch = np.array([self.get_edge_obs(n) for n in edge_names])
            actions, _ = self.edge_policy.predict(obs_batch, deterministic=True)
            for i, name in enumerate(edge_names):
                self.apply_edge_action(name, actions[i][0])

        # Central agent
        if exclude_agent != "central" and self.central_policy is not None:
            obs = self.get_central_obs()
            action, _ = self.central_policy.predict(obs, deterministic=True)
            self.apply_central_action(action)

        self.step_count += 1
