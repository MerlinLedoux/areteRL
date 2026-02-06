import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import envs  # noqa: F401 – declenche gymnasium.register
import gymnasium

from config import MODEL_PATH, N_EVAL_EPISODES


def evaluate(model, env, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return rewards


def plot_rewards(rewards):
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.6, label="Reward par episode")

    window = min(10, len(rewards))
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            avg,
            color="red",
            linewidth=2,
            label=f"Moyenne glissante ({window} eps)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward cumule")
    ax.set_title("Performance de l'agent PPO sur GridWorld")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150)
    print("Courbe sauvegardee dans reward_curve.png")
    plt.show()


def main():
    env = gymnasium.make("GridWorldContinuous-v0")
    model = PPO.load(MODEL_PATH)

    print(f"Evaluation sur {N_EVAL_EPISODES} episodes...")
    rewards = evaluate(model, env, N_EVAL_EPISODES)

    print(f"Reward moyen : {np.mean(rewards):.3f}")
    print(f"Reward min   : {np.min(rewards):.3f}")
    print(f"Reward max   : {np.max(rewards):.3f}")
    print(f"Ecart-type   : {np.std(rewards):.3f}")

    plot_rewards(rewards)
    env.close()


if __name__ == "__main__":
    main()
