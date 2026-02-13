import numpy as np
import matplotlib.pyplot as plt
import torch

from envs.grid_env import MultiAgentGridEnv
from models.ppo import ActorCritic
from config import (
    MODEL_PATH, N_EVAL_EPISODES, HIDDEN_SIZES,
    AGENT_NAMES, N_AGENTS, GOAL_POS,
)


def evaluate_episode(policy, env, device):
    """Run one episode. Returns per-agent rewards and success flags."""
    observations, _ = env.reset()
    agent_rewards = {name: 0.0 for name in AGENT_NAMES}
    agent_success = {name: False for name in AGENT_NAMES}

    while env.agents:
        active = list(env.agents)
        obs_array = np.stack([observations[a] for a in active])
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).to(device)

        with torch.no_grad():
            features = policy.shared(obs_tensor)
            actions = policy.actor_mean(features)
            actions = actions.cpu().numpy().clip(-1.0, 1.0)

        action_dict = {agent: actions[i] for i, agent in enumerate(active)}
        observations, rewards, terminations, truncations, infos = env.step(action_dict)

        for agent in action_dict:
            agent_rewards[agent] += rewards[agent]
            if terminations[agent]:
                agent_success[agent] = True

    return agent_rewards, agent_success


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentGridEnv()
    policy = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=HIDDEN_SIZES).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    policy.eval()

    all_rewards = {name: [] for name in AGENT_NAMES}
    all_success = {name: [] for name in AGENT_NAMES}
    joint_success = []

    print(f"Evaluation sur {N_EVAL_EPISODES} episodes...\n")

    for _ in range(N_EVAL_EPISODES):
        agent_rewards, agent_success = evaluate_episode(policy, env, device)
        for name in AGENT_NAMES:
            all_rewards[name].append(agent_rewards[name])
            all_success[name].append(agent_success[name])
        joint_success.append(all(agent_success.values()))

    print("=" * 60)
    print("Resultats par agent")
    print("=" * 60)

    for i, name in enumerate(AGENT_NAMES):
        rews = all_rewards[name]
        rate = np.mean(all_success[name]) * 100
        print(
            f"  {name} -> goal {GOAL_POS[i]}: "
            f"reward={np.mean(rews):>7.2f} +/- {np.std(rews):.2f} | "
            f"success={rate:.1f}%"
        )

    joint_rate = np.mean(joint_success) * 100
    print(f"\n  Joint success (les 5 ont fini): {joint_rate:.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, name in enumerate(AGENT_NAMES):
        ax = axes[i]
        rews = all_rewards[name]
        ax.plot(rews, alpha=0.6, color=colors[i])
        window = min(10, len(rews))
        if len(rews) >= window:
            avg = np.convolve(rews, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rews)), avg, color="black", linewidth=2)
        rate = np.mean(all_success[name]) * 100
        ax.set_title(f"{name} -> {GOAL_POS[i]} ({rate:.0f}%)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.bar(["Joint\nSuccess"], [joint_rate], color="green", alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Joint Success Rate")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Multi-Agent PPO Evaluation", fontsize=14)
    plt.tight_layout()
    plt.savefig("eval_rewards.png", dpi=150)
    print("\nPlot sauvegarde dans eval_rewards.png")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
