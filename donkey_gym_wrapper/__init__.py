from gym.envs.registration import register


register(
    id='donkey-vae-v0',
    entry_point='donkey_gym_wrapper.env:DonkeyVAEEnv',
    timestep_limit=2000,
)
