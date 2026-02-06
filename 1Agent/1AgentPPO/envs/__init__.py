import gymnasium

gymnasium.register(
    id="GridWorldContinuous-v0",
    entry_point="envs.grid_env:GridWorldEnv",
)
