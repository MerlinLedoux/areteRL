import os
import time
import numpy as np
import torch

from envs.grid_env import MultiAgentGridEnv
from models.ppo import ActorCritic, RolloutBuffer, compute_gae, ppo_update
from config import (
    PPO_CONFIG, HIDDEN_SIZES, TOTAL_TIMESTEPS,
    SAVE_INTERVAL, MODEL_DIR, MODEL_PATH,
    AGENT_NAMES,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = MultiAgentGridEnv()
    policy = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=HIDDEN_SIZES).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=PPO_CONFIG["learning_rate"])

    os.makedirs(MODEL_DIR, exist_ok=True)

    n_steps = PPO_CONFIG["n_steps"]
    gamma = PPO_CONFIG["gamma"]
    gae_lam = PPO_CONFIG["gae_lambda"]

    global_step = 0
    num_episodes = 0
    start_time = time.time()

    print(f"Training for {TOTAL_TIMESTEPS} total agent-steps")
    print(f"Collecting {n_steps} steps/agent per rollout")

    while global_step < TOTAL_TIMESTEPS:
        # ---- Phase 1 : Collect rollout ----
        agent_data = {
            name: {"obs": [], "act": [], "logp": [], "rew": [], "val": [], "done": []}
            for name in AGENT_NAMES
        }
        steps_collected = {name: 0 for name in AGENT_NAMES}
        rollout_episode_rewards = []

        observations, _ = env.reset()
        current_episode_reward = 0.0

        while min(steps_collected.values()) < n_steps:
            active_agents = list(env.agents)

            if not active_agents:
                rollout_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                num_episodes += 1
                observations, _ = env.reset()
                continue

            obs_array = np.stack([observations[a] for a in active_agents])
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).to(device)

            with torch.no_grad():
                actions_t, log_probs_t, values_t = policy.get_action(obs_tensor)

            actions_raw = actions_t.cpu().numpy()
            actions_clipped = actions_raw.clip(-1.0, 1.0)
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            action_dict = {
                agent: actions_clipped[i] for i, agent in enumerate(active_agents)
            }

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            for i, agent in enumerate(active_agents):
                done = terminations[agent] or truncations[agent]
                agent_data[agent]["obs"].append(obs_array[i])
                agent_data[agent]["act"].append(actions_raw[i])
                agent_data[agent]["logp"].append(log_probs_np[i])
                agent_data[agent]["rew"].append(rewards[agent])
                agent_data[agent]["val"].append(values_np[i])
                agent_data[agent]["done"].append(done)
                steps_collected[agent] += 1
                current_episode_reward += rewards[agent]

            observations = next_obs

            if not env.agents:
                rollout_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                num_episodes += 1
                observations, _ = env.reset()

        # ---- Phase 2 : Compute GAE per agent, then flatten ----
        all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []

        for agent in AGENT_NAMES:
            d = agent_data[agent]
            T = len(d["rew"])
            if T == 0:
                continue

            obs_arr = np.array(d["obs"])
            act_arr = np.array(d["act"])
            logp_arr = np.array(d["logp"])
            rew_arr = np.array(d["rew"])
            val_arr = np.array(d["val"])
            done_arr = np.array(d["done"])

            if done_arr[-1]:
                last_value = 0.0
            else:
                if agent in observations:
                    last_obs_t = torch.as_tensor(
                        observations[agent], dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        last_value = policy.get_value(last_obs_t).item()
                else:
                    last_value = 0.0

            advantages, returns = compute_gae(
                rew_arr, val_arr, done_arr, last_value, gamma, gae_lam
            )

            n = min(T, n_steps)
            all_obs.append(obs_arr[:n])
            all_act.append(act_arr[:n])
            all_logp.append(logp_arr[:n])
            all_adv.append(advantages[:n])
            all_ret.append(returns[:n])

        all_obs = np.concatenate(all_obs)
        all_act = np.concatenate(all_act)
        all_logp = np.concatenate(all_logp)
        all_adv = np.concatenate(all_adv)
        all_ret = np.concatenate(all_ret)

        global_step += len(all_obs)

        # ---- Phase 3 : PPO update ----
        buffer = RolloutBuffer()
        buffer.finalize(all_obs, all_act, all_logp, all_adv, all_ret)

        metrics = ppo_update(policy, buffer, optimizer, PPO_CONFIG, device)

        # ---- Logging ----
        elapsed = time.time() - start_time
        fps = global_step / max(elapsed, 1e-6)
        mean_rew = np.mean(rollout_episode_rewards) if rollout_episode_rewards else float("nan")

        print(
            f"Step {global_step:>8d}/{TOTAL_TIMESTEPS} | "
            f"Ep: {num_episodes:>5d} | "
            f"Mean reward: {mean_rew:>8.2f} | "
            f"PG: {metrics['pg_loss']:.4f} | "
            f"VF: {metrics['vf_loss']:.4f} | "
            f"Ent: {metrics['entropy']:.4f} | "
            f"FPS: {fps:.0f}"
        )

        # ---- Checkpoint ----
        if global_step % SAVE_INTERVAL < (n_steps * len(AGENT_NAMES)):
            path = os.path.join(MODEL_DIR, f"ppo_multi_{global_step}.pt")
            torch.save(policy.state_dict(), path)
            print(f"  -> Checkpoint: {path}")

    # ---- Final save ----
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_PATH}")
    print(f"Total episodes: {num_episodes}")
    print(f"Wall time: {time.time() - start_time:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
