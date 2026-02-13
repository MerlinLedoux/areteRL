import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden_sizes=(64, 64)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.shared = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(in_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def get_action(self, obs):
        """Returns action, log_prob, value for a batch of observations."""
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, value

    def evaluate(self, obs, actions):
        """Returns log_prob, entropy, value for stored (obs, action) pairs."""
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return log_prob, entropy, value

    def get_value(self, obs):
        """Returns value estimate only (for bootstrap)."""
        features = self.shared(obs)
        return self.critic(features).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute GAE advantages and returns for a single agent trajectory."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Stores flattened transitions after per-agent GAE computation."""

    def __init__(self):
        self._size = 0
        self.observations = None
        self.actions = None
        self.log_probs = None
        self.advantages = None
        self.returns = None

    def finalize(self, obs, act, logp, adv, ret):
        self.observations = torch.as_tensor(obs, dtype=torch.float32)
        self.actions = torch.as_tensor(act, dtype=torch.float32)
        self.log_probs = torch.as_tensor(logp, dtype=torch.float32)
        self.advantages = torch.as_tensor(adv, dtype=torch.float32)
        self.returns = torch.as_tensor(ret, dtype=torch.float32)
        self._size = len(obs)

    def size(self):
        return self._size

    def get_batches(self, batch_size, device="cpu"):
        indices = np.random.permutation(self._size)
        for start in range(0, self._size, batch_size):
            end = min(start + batch_size, self._size)
            idx = indices[start:end]
            yield (
                self.observations[idx].to(device),
                self.actions[idx].to(device),
                self.log_probs[idx].to(device),
                self.advantages[idx].to(device),
                self.returns[idx].to(device),
            )


def ppo_update(policy, buffer, optimizer, config, device="cpu"):
    """Perform PPO clipped surrogate update. Returns dict of metrics."""
    clip_range = config["clip_range"]
    ent_coef = config["ent_coef"]
    vf_coef = config["vf_coef"]
    max_grad_norm = config["max_grad_norm"]
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]

    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        for obs_b, act_b, old_logp_b, adv_b, ret_b in buffer.get_batches(
            batch_size, device
        ):
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            new_logp, entropy, values = policy.evaluate(obs_b, act_b)

            ratio = (new_logp - old_logp_b).exp()
            pg_loss1 = -adv_b * ratio
            pg_loss2 = -adv_b * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            vf_loss = 0.5 * ((values - ret_b) ** 2).mean()

            entropy_loss = -entropy.mean()

            loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "pg_loss": total_pg_loss / max(n_updates, 1),
        "vf_loss": total_vf_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }
