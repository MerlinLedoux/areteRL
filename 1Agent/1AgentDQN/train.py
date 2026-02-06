import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import envs  # noqa: F401 – declenche gymnasium.register
import gymnasium

from config import DQN_CONFIG, TOTAL_TIMESTEPS, MODEL_PATH


def main():
    env = gymnasium.make("GridWorld-v0")
    env = Monitor(env)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        **DQN_CONFIG,
    )

    print(f"Entrainement DQN pour {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()
