import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from stable_baselines3 import DQN

import envs  # noqa: F401
import gymnasium

from config import MODEL_PATH, GRID_SIZE, GOAL_POS, GOAL_RADIUS

REF_POINTS = np.array([(1, 0), (0, 1), (2, 1), (1, 2)], dtype=np.float32)
REF_LABELS = ["(1,0)", "(0,1)", "(2,1)", "(1,2)"]


def collect_episode(model, env):
    positions = []
    rewards = []
    obs, _ = env.reset()
    positions.append(env.unwrapped.agent_pos.copy())
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        positions.append(env.unwrapped.agent_pos.copy())
        rewards.append(reward)
        done = terminated or truncated
    return np.array(positions), np.array(rewards)


def visualize(positions, rewards):
    cum_rewards = np.concatenate(([0.0], np.cumsum(rewards)))

    fig, ax = plt.subplots(figsize=(7, 7))

    goal = np.array(GOAL_POS, dtype=np.float32)
    goal_circle = patches.Circle(
        goal, GOAL_RADIUS, color="green", alpha=0.3, label="Zone objectif"
    )
    ax.add_patch(goal_circle)
    ax.plot(*goal, "g+", markersize=12, markeredgewidth=2)

    for pt, label in zip(REF_POINTS, REF_LABELS):
        ax.plot(*pt, "ks", markersize=8)
        ax.annotate(label, pt, textcoords="offset points", xytext=(5, 5), fontsize=9)

    trajectory, = ax.plot([], [], "b-", alpha=0.3, linewidth=1, label="Trajectoire")
    agent_dot, = ax.plot([], [], "ro", markersize=10, label="Agent")
    segments = [ax.plot([], [], "r--", alpha=0.5, linewidth=1)[0] for _ in REF_POINTS]

    dist_texts = []
    for _ in REF_POINTS:
        txt = ax.text(0, 0, "", fontsize=8, color="red", ha="center")
        dist_texts.append(txt)

    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    ax.set_xlim(-0.15, GRID_SIZE + 0.15)
    ax.set_ylim(-0.15, GRID_SIZE + 0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Agent DQN — Navigation GridWorld")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    rect = patches.Rectangle((0, 0), GRID_SIZE, GRID_SIZE, linewidth=2,
                              edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    reward_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5)
    )

    def init():
        trajectory.set_data([], [])
        agent_dot.set_data([], [])
        for seg in segments:
            seg.set_data([], [])
        for txt in dist_texts:
            txt.set_text("")
        step_text.set_text("")
        reward_text.set_text("")
        return [trajectory, agent_dot, step_text, reward_text] + segments + dist_texts

    def update(frame):
        pos = positions[frame]

        trajectory.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])
        agent_dot.set_data([pos[0]], [pos[1]])

        for i, (seg, ref, txt) in enumerate(zip(segments, REF_POINTS, dist_texts)):
            seg.set_data([pos[0], ref[0]], [pos[1], ref[1]])
            mid = (pos + ref) / 2
            dist = np.linalg.norm(pos - ref)
            txt.set_position(mid)
            txt.set_text(f"{dist:.2f}")

        step_text.set_text(f"Step {frame}/{len(positions) - 1}")
        reward_text.set_text(f"Reward = {cum_rewards[frame]:.3f}")

        return [trajectory, agent_dot, step_text, reward_text] + segments + dist_texts

    anim = FuncAnimation(
        fig, update, frames=len(positions),
        init_func=init, interval=150, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()


def main():
    env = gymnasium.make("GridWorld-v0")
    model = DQN.load(MODEL_PATH)

    print("Collecte d'un episode...")
    positions, rewards = collect_episode(model, env)
    print(f"Episode de {len(positions) - 1} steps")
    print(f"Reward total : {rewards.sum():.3f}")

    visualize(positions, rewards)
    env.close()


if __name__ == "__main__":
    main()
