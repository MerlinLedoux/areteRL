"""
Controle manuel des 5 agents au clavier.
Touches : z/q/s/d = haut/gauche/bas/droite, espace = ne rien faire
Affiche obs, reward detail et positions a chaque step.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from envs.grid_env import MultiAgentGridEnv
from config import GRID_SIZE, GOAL_POS, GOAL_RADIUS, REF_POINTS, AGENT_NAMES

AGENT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# Mapping clavier -> action (dx, dy)
KEY_MAP = {
    "z": np.array([0.0, 1.0]),   # haut
    "s": np.array([0.0, -1.0]),  # bas
    "d": np.array([1.0, 0.0]),   # droite
    "q": np.array([-1.0, 0.0]),  # gauche
    " ": np.array([0.0, 0.0]),   # ne rien faire
}


def draw_grid(env):
    """Dessine la grille avec les positions actuelles."""
    fig, ax = plt.subplots(figsize=(7, 7))

    rect = patches.Rectangle((0, 0), GRID_SIZE, GRID_SIZE, linewidth=2,
                              edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    # Points de reference (coins)
    for pt in REF_POINTS:
        ax.plot(*pt, "ks", markersize=8, alpha=0.3)
        ax.annotate(f"({pt[0]},{pt[1]})", pt, textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.5)

    # Goals et agents
    for i, name in enumerate(AGENT_NAMES):
        goal = np.array(GOAL_POS[i])
        circle = patches.Circle(goal, GOAL_RADIUS, color=AGENT_COLORS[i], alpha=0.2)
        ax.add_patch(circle)
        ax.plot(*goal, "+", color=AGENT_COLORS[i], markersize=12, markeredgewidth=2)

        if name in env._positions:
            pos = env._positions[name]
            done = env._done_flags.get(name, False)
            marker = "x" if done else "o"
            ax.plot(pos[0], pos[1], marker, color=AGENT_COLORS[i],
                    markersize=12, markeredgewidth=2, label=f"{name} {'DONE' if done else ''}")
            ax.annotate(name[-1], (pos[0], pos[1]), textcoords="offset points",
                        xytext=(-5, 8), fontsize=9, fontweight="bold",
                        color=AGENT_COLORS[i])

    ax.set_xlim(-0.3, GRID_SIZE + 0.3)
    ax.set_ylim(-0.3, GRID_SIZE + 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"Step {env._steps}")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return fig


def get_action_for_agent(agent):
    """Demande une touche au joueur pour un agent."""
    while True:
        key = input(f"  {agent} [z/q/s/d/espace]: ").strip().lower()
        if key == "":
            key = " "
        if key in KEY_MAP:
            return KEY_MAP[key]
        print(f"    Touche invalide '{key}'. Utilise z/q/s/d ou espace.")


def print_step_detail(agent, action, obs, reward_parts, reward_total, dist_to_goal, pos, terminated):
    """Affiche le detail d'un step pour un agent."""
    goal = GOAL_POS[AGENT_NAMES.index(agent)]
    print(f"\n  --- {agent} -> goal {goal} ---")
    print(f"    Position:  ({pos[0]:.3f}, {pos[1]:.3f})")
    print(f"    Action:    ({action[0]:+.2f}, {action[1]:+.2f})")
    print(f"    Obs:       [{', '.join(f'{v:+.3f}' for v in obs)}]")
    print(f"    Dist goal: {dist_to_goal:.3f}")
    print(f"    Rewards:")
    for name, val in reward_parts.items():
        print(f"      {name:20s}: {val:+.4f}")
    print(f"      {'TOTAL':20s}: {reward_total:+.4f}")
    if terminated:
        print(f"    >>> GOAL ATTEINT <<<")


def main():
    env = MultiAgentGridEnv()
    observations, _ = env.reset(seed=42)

    print("=" * 60)
    print("CONTROLE MANUEL - 5 AGENTS")
    print("Touches: z=haut, s=bas, q=gauche, d=droite, espace=rien")
    print("Ctrl+C pour quitter")
    print("=" * 60)

    # Afficher etat initial
    print("\n--- ETAT INITIAL ---")
    for name in AGENT_NAMES:
        pos = env._positions[name]
        goal = env._goal_map[name]
        dist = np.linalg.norm(pos - goal)
        print(f"  {name}: pos=({pos[0]:.2f}, {pos[1]:.2f}) goal=({goal[0]:.1f}, {goal[1]:.1f}) dist={dist:.3f}")
        print(f"    obs=[{', '.join(f'{v:+.3f}' for v in observations[name])}]")

    fig = draw_grid(env)
    cumulative_rewards = {name: 0.0 for name in AGENT_NAMES}

    try:
        while env.agents:
            print(f"\n{'='*60}")
            print(f"STEP {env._steps + 1} - Agents actifs: {env.agents}")
            print(f"{'='*60}")

            # Collecter les actions
            actions = {}
            for agent in env.agents:
                actions[agent] = get_action_for_agent(agent)

            # Accumuler les resultats sur les 5 repetitions
            all_observations = {}
            all_rewards = {a: 0.0 for a in actions}
            all_terminations = {a: False for a in actions}
            all_truncations = {a: False for a in actions}
            all_infos = {}

            for rep in range(5):
                step_actions = {a: actions[a] for a in env.agents if a in actions}
                if not step_actions:
                    break
                observations, rewards, terminations, truncations, infos = env.step(step_actions)
                for a in step_actions:
                    all_observations[a] = observations[a]
                    all_rewards[a] += rewards[a]
                    all_infos[a] = infos[a]
                    if terminations[a]:
                        all_terminations[a] = True
                    if truncations[a]:
                        all_truncations[a] = True
                if not env.agents:
                    break

            # Afficher le detail pour chaque agent
            for agent in list(actions.keys()):
                action = actions[agent]
                obs = all_observations[agent]
                pos = env._positions[agent]
                dist_to_goal = all_infos[agent]["dist_to_goal"]
                terminated = all_terminations[agent]

                # Decomposer le reward
                raw_action = np.clip(action, -1.0, 1.0)
                forbidden = 0.0
                if agent in ("agent_1", "agent_3"):
                    forbidden = -abs(float(raw_action[0])) * 0.5
                elif agent in ("agent_2", "agent_4"):
                    forbidden = -abs(float(raw_action[1])) * 0.5

                distance_pen = -dist_to_goal * 0.1

                norm = np.linalg.norm(raw_action)
                angle_rew = 0.0
                if norm > 1e-8 and dist_to_goal > 1e-8:
                    direction = raw_action / norm
                    if agent in ("agent_1", "agent_3"):
                        direction[0] = 0.0
                    if agent in ("agent_2", "agent_4"):
                        direction[1] = 0.0
                    goal = env._goal_map[agent]
                    vec = goal - pos
                    goal_dir = vec / np.linalg.norm(vec)
                    angle_rew = float(np.dot(direction, goal_dir)) * 0.05

                parts = {
                    "forbidden_penalty": forbidden,
                    "distance_penalty": distance_pen,
                    "angle_reward": angle_rew,
                    "time_penalty": -0.01,
                }
                if terminated:
                    parts["goal_reached"] = 5.0
                if all_rewards[agent] > sum(parts.values()) * 5 + 0.01:
                    parts["all_done_bonus"] = 20.0

                cumulative_rewards[agent] += all_rewards[agent]

                print_step_detail(agent, action, obs, parts, all_rewards[agent],
                                  dist_to_goal, pos, terminated)

            # Afficher les rewards cumules
            print(f"\n  Rewards cumules:")
            for name in AGENT_NAMES:
                done = env._done_flags.get(name, False)
                status = " [DONE]" if done else ""
                print(f"    {name}: {cumulative_rewards[name]:+.3f}{status}")

            # Mettre a jour le dessin
            plt.close(fig)
            fig = draw_grid(env)

        # Fin
        print(f"\n{'='*60}")
        print("EPISODE TERMINEE")
        print(f"{'='*60}")
        print(f"Steps: {env._steps}")
        for name in AGENT_NAMES:
            print(f"  {name}: reward total = {cumulative_rewards[name]:+.3f}")
        print(f"  TEAM TOTAL: {sum(cumulative_rewards.values()):+.3f}")

        plt.close(fig)

    except KeyboardInterrupt:
        print("\n\nInterrompu.")
        plt.close("all")

    env.close()


if __name__ == "__main__":
    main()
