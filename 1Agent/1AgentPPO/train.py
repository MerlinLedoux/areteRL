import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import envs  # noqa: F401 – declenche gymnasium.register
import gymnasium

from config import PPO_CONFIG, TOTAL_TIMESTEPS, MODEL_PATH


def main():
    env = gymnasium.make("GridWorldContinuous-v0")
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        **PPO_CONFIG,
    )

    print(f"Entrainement PPO pour {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()
