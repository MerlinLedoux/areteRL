import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs import MultiAgentWorld, EdgeAgentEnv, CentralAgentEnv
from config import (
    PPO_EDGE_CONFIG, PPO_CENTRAL_CONFIG,
    TIMESTEPS_PER_ROUND, NUM_TRAINING_ROUNDS,
    MODEL_PATH_EDGE, MODEL_PATH_CENTRAL, EDGE_AGENTS,
)


def make_edge_vec_env(edge_policy, central_policy):
    def make_env(agent_name):
        def _init():
            world = MultiAgentWorld()
            world.edge_policy = edge_policy
            world.central_policy = central_policy
            return Monitor(EdgeAgentEnv(world, agent_name))
        return _init

    env_fns = [make_env(name) for name in EDGE_AGENTS]
    return DummyVecEnv(env_fns)


def make_central_vec_env(edge_policy, central_policy, n_envs=4):
    def make_env():
        def _init():
            world = MultiAgentWorld()
            world.edge_policy = edge_policy
            world.central_policy = central_policy
            return Monitor(CentralAgentEnv(world))
        return _init

    return DummyVecEnv([make_env() for _ in range(n_envs)])


def main():
    os.makedirs(os.path.dirname(MODEL_PATH_EDGE), exist_ok=True)

    edge_policy = None
    central_policy = None

    for round_idx in range(NUM_TRAINING_ROUNDS):
        print(f"\n{'=' * 60}")
        print(f"  ROUND {round_idx + 1}/{NUM_TRAINING_ROUNDS}")
        print(f"{'=' * 60}")

        # --- Phase 1 : entrainer les agents d'arete (central gele) ---
        print(f"\n--- Training EDGE agents ---")
        edge_vec_env = make_edge_vec_env(edge_policy, central_policy)

        if edge_policy is None:
            edge_model = PPO("MlpPolicy", edge_vec_env, verbose=1, device="cuda", **PPO_EDGE_CONFIG)
        else:
            edge_model = PPO.load(MODEL_PATH_EDGE, env=edge_vec_env, device="cuda")

        edge_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, reset_num_timesteps=False)
        edge_model.save(MODEL_PATH_EDGE)
        edge_policy = edge_model
        edge_vec_env.close()

        # --- Phase 2 : entrainer l'agent central (aretes gelees) ---
        print(f"\n--- Training CENTRAL agent ---")
        central_vec_env = make_central_vec_env(edge_policy, central_policy)

        if central_policy is None:
            central_model = PPO("MlpPolicy", central_vec_env, verbose=1, device="cuda", **PPO_CENTRAL_CONFIG)
        else:
            central_model = PPO.load(MODEL_PATH_CENTRAL, env=central_vec_env, device="cuda")

        central_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, reset_num_timesteps=False)
        central_model.save(MODEL_PATH_CENTRAL)
        central_policy = central_model
        central_vec_env.close()

    print(f"\nTraining complete.")
    print(f"  Edge model:    {MODEL_PATH_EDGE}")
    print(f"  Central model: {MODEL_PATH_CENTRAL}")


if __name__ == "__main__":
    main()
