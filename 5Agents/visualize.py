import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

from envs import MultiAgentWorld
from config import (
    MODEL_PATH_EDGE, MODEL_PATH_CENTRAL,
    GRID_SIZE, GOAL_POS_CENTRAL, GOAL_RADIUS_CENTRAL,
    MAX_STEPS_PER_EPISODE, EDGE_AGENTS, GOAL_RADIUS_EDGE,
)

EDGE_COLORS = {
    "bottom": "blue",
    "left": "orange",
    "right": "purple",
    "top": "cyan",
}


def collect_joint_episode(edge_model, central_model):
    world = MultiAgentWorld()
    world.reset()
    goal = np.array(GOAL_POS_CENTRAL, dtype=np.float32)

    history = {
        "central": [world.central_pos.copy()],
    }
    for name in EDGE_AGENTS:
        history[name] = [world.get_edge_2d_pos(name).copy()]

    rewards = []

    for step in range(MAX_STEPS_PER_EPISODE):
        central_obs = world.get_central_obs()
        central_action, _ = central_model.predict(central_obs, deterministic=True)
        world.apply_central_action(central_action)

        names = list(EDGE_AGENTS.keys())
        obs_batch = np.array([world.get_edge_obs(n) for n in names])
        actions, _ = edge_model.predict(obs_batch, deterministic=True)
        for i, name in enumerate(names):
            world.apply_edge_action(name, actions[i][0])

        history["central"].append(world.central_pos.copy())
        for name in EDGE_AGENTS:
            history[name].append(world.get_edge_2d_pos(name).copy())

        # Reward central
        obs_after = world.get_central_obs()
        dist = np.linalg.norm(world.central_pos - goal)
        if dist <= GOAL_RADIUS_CENTRAL:
            rewards.append(10.0)
            break
        else:
            rewards.append(-0.01 + (-np.sum(np.abs(obs_after - 1.0))))

    for key in history:
        history[key] = np.array(history[key])

    return history, np.array(rewards)


def visualize(history, rewards):
    cum_rewards = np.concatenate(([0.0], np.cumsum(rewards)))
    n_frames = len(history["central"])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cadre
    rect = patches.Rectangle((0, 0), GRID_SIZE, GRID_SIZE, linewidth=2,
                              edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    # Goal central
    goal = np.array(GOAL_POS_CENTRAL, dtype=np.float32)
    goal_circle = patches.Circle(goal, GOAL_RADIUS_CENTRAL, color="green", alpha=0.3)
    ax.add_patch(goal_circle)
    ax.plot(*goal, "g+", markersize=12, markeredgewidth=2)

    # Goals aretes (milieu de chaque arete)
    edge_goals = {
        "bottom": (1.0, 0.0), "left": (0.0, 1.0),
        "right": (2.0, 1.0), "top": (1.0, 2.0),
    }
    for name, pos in edge_goals.items():
        ax.plot(*pos, "x", color=EDGE_COLORS[name], markersize=10, markeredgewidth=2)

    # Trajectoire et point central
    central_traj, = ax.plot([], [], "r-", alpha=0.3, linewidth=1)
    central_dot, = ax.plot([], [], "ro", markersize=10, label="Central")

    # Trajectoires et points aretes
    edge_dots = {}
    edge_trajs = {}
    segments = {}
    dist_texts = {}
    for name in EDGE_AGENTS:
        color = EDGE_COLORS[name]
        traj, = ax.plot([], [], "-", color=color, alpha=0.3, linewidth=1)
        dot, = ax.plot([], [], "o", color=color, markersize=8, label=name.capitalize())
        seg, = ax.plot([], [], "--", color=color, alpha=0.4, linewidth=1)
        txt = ax.text(0, 0, "", fontsize=7, color=color, ha="center")
        edge_trajs[name] = traj
        edge_dots[name] = dot
        segments[name] = seg
        dist_texts[name] = txt

    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    reward_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5)
    )

    ax.set_xlim(-0.2, GRID_SIZE + 0.2)
    ax.set_ylim(-0.2, GRID_SIZE + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Agent PPO — 5 Agents sur GridWorld")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    all_artists = ([central_traj, central_dot, step_text, reward_text]
                   + list(edge_trajs.values()) + list(edge_dots.values())
                   + list(segments.values()) + list(dist_texts.values()))

    def init():
        central_traj.set_data([], [])
        central_dot.set_data([], [])
        for name in EDGE_AGENTS:
            edge_trajs[name].set_data([], [])
            edge_dots[name].set_data([], [])
            segments[name].set_data([], [])
            dist_texts[name].set_text("")
        step_text.set_text("")
        reward_text.set_text("")
        return all_artists

    def update(frame):
        c_pos = history["central"][frame]
        central_traj.set_data(history["central"][:frame + 1, 0],
                              history["central"][:frame + 1, 1])
        central_dot.set_data([c_pos[0]], [c_pos[1]])

        for name in EDGE_AGENTS:
            e_pos = history[name][frame]
            edge_trajs[name].set_data(history[name][:frame + 1, 0],
                                      history[name][:frame + 1, 1])
            edge_dots[name].set_data([e_pos[0]], [e_pos[1]])
            segments[name].set_data([c_pos[0], e_pos[0]], [c_pos[1], e_pos[1]])
            d = np.linalg.norm(c_pos - e_pos)
            mid = (c_pos + e_pos) / 2
            dist_texts[name].set_position(mid)
            dist_texts[name].set_text(f"{d:.2f}")

        step_text.set_text(f"Step {frame}/{n_frames - 1}")
        reward_text.set_text(f"Reward = {cum_rewards[frame]:.3f}")

        if frame == n_frames - 1:
            timer = fig.canvas.new_timer(interval=1000)
            timer.add_callback(lambda: plt.close(fig))
            timer.single_shot = True
            timer.start()
            fig._close_timer = timer

        return all_artists

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=150, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()


def main():
    edge_model = PPO.load(MODEL_PATH_EDGE)
    central_model = PPO.load(MODEL_PATH_CENTRAL)

    for k in range(10):
        print(f"\n--- Episode {k + 1}/10 ---")
        history, rewards = collect_joint_episode(edge_model, central_model)
        print(f"Steps: {len(history['central']) - 1} | Reward total: {rewards.sum():.3f}")
        visualize(history, rewards)


if __name__ == "__main__":
    main()
