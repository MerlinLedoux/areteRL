import gymnasium

gymnasium.register(
    id="GridWorld-v0",
    entry_point="envs.grid_env:GridWorldEnv",
)
