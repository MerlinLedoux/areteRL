import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs import MultiAgentWorld
from config import (
    MODEL_PATH_EDGE, MODEL_PATH_CENTRAL,
    N_EVAL_EPISODES, MAX_STEPS_PER_EPISODE,
    EDGE_AGENTS, GOAL_RADIUS_CENTRAL, GOAL_RADIUS_EDGE,
    GOAL_POS_CENTRAL, TIME_PENALTY, REWARD_GOAL,
)


def run_joint_episode(edge_model, central_model):
    world = MultiAgentWorld()
    world.reset()
    goal = np.array(GOAL_POS_CENTRAL, dtype=np.float32)

    total_central_reward = 0.0
    total_edge_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Central agent
        central_obs = world.get_central_obs()
        central_action, _ = central_model.predict(central_obs, deterministic=True)
        world.apply_central_action(central_action)

        # Edge agents (batched)
        names = list(EDGE_AGENTS.keys())
        obs_batch = np.array([world.get_edge_obs(n) for n in names])
        actions, _ = edge_model.predict(obs_batch, deterministic=True)
        for i, name in enumerate(names):
            world.apply_edge_action(name, actions[i][0])

        # Central reward
        desired = 1.0
        obs_after = world.get_central_obs()
        dist_to_goal = np.linalg.norm(world.central_pos - goal)
        if dist_to_goal <= GOAL_RADIUS_CENTRAL:
            total_central_reward += REWARD_GOAL
            break
        else:
            total_central_reward += TIME_PENALTY + (-np.sum(np.abs(obs_after - desired)))

        # Edge reward
        for name in names:
            pos = world.edge_positions[name]
            dist_to_mid = abs(pos - 1.0)
            if dist_to_mid <= GOAL_RADIUS_EDGE:
                total_edge_reward += REWARD_GOAL
            else:
                total_edge_reward += -dist_to_mid + TIME_PENALTY

    return total_central_reward, total_edge_reward


def main():
    edge_model = PPO.load(MODEL_PATH_EDGE)
    central_model = PPO.load(MODEL_PATH_CENTRAL)

    central_rewards = []
    edge_rewards = []

    for ep in range(N_EVAL_EPISODES):
        cr, er = run_joint_episode(edge_model, central_model)
        central_rewards.append(cr)
        edge_rewards.append(er)

    print(f"--- Central agent ({N_EVAL_EPISODES} episodes) ---")
    print(f"  Reward moyen : {np.mean(central_rewards):.3f}")
    print(f"  Reward min   : {np.min(central_rewards):.3f}")
    print(f"  Reward max   : {np.max(central_rewards):.3f}")

    print(f"\n--- Edge agents ({N_EVAL_EPISODES} episodes) ---")
    print(f"  Reward moyen : {np.mean(edge_rewards):.3f}")
    print(f"  Reward min   : {np.min(edge_rewards):.3f}")
    print(f"  Reward max   : {np.max(edge_rewards):.3f}")

    # Plot
    episodes = np.arange(1, N_EVAL_EPISODES + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(episodes, central_rewards, alpha=0.6)
    ax1.set_title("Central agent — Reward par episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, edge_rewards, alpha=0.6, color="orange")
    ax2.set_title("Edge agents — Reward par episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eval_rewards.png", dpi=150)
    print("\nCourbe sauvegardee dans eval_rewards.png")
    plt.show()


if __name__ == "__main__":
    main()
