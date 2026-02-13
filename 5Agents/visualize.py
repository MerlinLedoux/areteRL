import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch

from envs.grid_env import MultiAgentGridEnv
from models.ppo import ActorCritic
from config import (
    MODEL_PATH, GRID_SIZE, GOAL_POS, GOAL_RADIUS,
    REF_POINTS, HIDDEN_SIZES, AGENT_NAMES,
)

AGENT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
REF_LABELS = [f"({p[0]},{p[1]})" for p in REF_POINTS]


def collect_episode(policy, env, device):
    """Collect one episode, return per-agent position histories and step rewards."""
    observations, _ = env.reset()

    history = {name: [env._positions[name].copy()] for name in AGENT_NAMES}
    step_rewards = []

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

        step_rewards.append(sum(rewards.values()))

        for name in AGENT_NAMES:
            if name in env._positions:
                history[name].append(env._positions[name].copy())
            else:
                history[name].append(history[name][-1])

    for name in history:
        history[name] = np.array(history[name])

    return history, np.array(step_rewards)


def visualize(history, rewards):
    cum_rewards = np.concatenate(([0.0], np.cumsum(rewards)))
    n_frames = max(len(h) for h in history.values())

    # Pad shorter histories
    for name in history:
        h = history[name]
        if len(h) < n_frames:
            pad = np.tile(h[-1], (n_frames - len(h), 1))
            history[name] = np.concatenate([h, pad])

    fig, ax = plt.subplots(figsize=(8, 8))

    rect = patches.Rectangle(
        (0, 0), GRID_SIZE, GRID_SIZE, linewidth=2,
        edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)

    ref_pts = np.array(REF_POINTS, dtype=np.float32)
    for pt, label in zip(ref_pts, REF_LABELS):
        ax.plot(*pt, "ks", markersize=8)
        ax.annotate(label, pt, textcoords="offset points", xytext=(5, 5), fontsize=8)

    for i, name in enumerate(AGENT_NAMES):
        goal = np.array(GOAL_POS[i], dtype=np.float32)
        circle = patches.Circle(goal, GOAL_RADIUS, color=AGENT_COLORS[i], alpha=0.2)
        ax.add_patch(circle)
        ax.plot(*goal, "+", color=AGENT_COLORS[i], markersize=10, markeredgewidth=2)

    trajs = {}
    dots = {}
    for i, name in enumerate(AGENT_NAMES):
        traj, = ax.plot([], [], "-", color=AGENT_COLORS[i], alpha=0.3, linewidth=1)
        dot, = ax.plot([], [], "o", color=AGENT_COLORS[i], markersize=8, label=name)
        trajs[name] = traj
        dots[name] = dot

    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    reward_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    ax.set_xlim(-0.2, GRID_SIZE + 0.2)
    ax.set_ylim(-0.2, GRID_SIZE + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Agent PPO - 5 Agents GridWorld")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    all_artists = list(trajs.values()) + list(dots.values()) + [step_text, reward_text]

    def init():
        for name in AGENT_NAMES:
            trajs[name].set_data([], [])
            dots[name].set_data([], [])
        step_text.set_text("")
        reward_text.set_text("")
        return all_artists

    def update(frame):
        for name in AGENT_NAMES:
            h = history[name]
            f = min(frame, len(h) - 1)
            pos = h[f]
            trajs[name].set_data(h[: f + 1, 0], h[: f + 1, 1])
            dots[name].set_data([pos[0]], [pos[1]])

        step_text.set_text(f"Step {frame}/{n_frames - 1}")
        r_idx = min(frame, len(cum_rewards) - 1)
        reward_text.set_text(f"Team Reward = {cum_rewards[r_idx]:.2f}")

        if frame == n_frames - 1:
            timer = fig.canvas.new_timer(interval=1500)
            timer.add_callback(lambda: plt.close(fig))
            timer.single_shot = True
            timer.start()
            fig._close_timer = timer

        return all_artists

    _anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=150, blit=True, repeat=False,
    )

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentGridEnv()
    policy = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=HIDDEN_SIZES).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    policy.eval()

    for k in range(10):
        print(f"\n--- Episode {k + 1}/10 ---")
        history, rewards = collect_episode(policy, env, device)
        total_steps = max(len(h) for h in history.values()) - 1
        print(f"Steps: {total_steps} | Team reward: {rewards.sum():.2f}")
        for i, name in enumerate(AGENT_NAMES):
            n = len(history[name]) - 1
            print(f"  {name} -> {GOAL_POS[i]}: {n} steps")
        visualize(history, rewards)

    env.close()


if __name__ == "__main__":
    main()
