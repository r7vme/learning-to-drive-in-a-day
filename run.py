#!/usr/bin/env python
# Copyright (c) 2018 Roma Sokolkov
# MIT License

import os
import gym
import numpy as np

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController

# Registers donkey-vae-v0 gym env.
import donkey_gym_wrapper

env = gym.make('donkey-vae-v0')

PATH_MODEL_VAE = "vae.json"
# Final filename will be PATH_MODEL_DDPG + ".pkl"
PATH_MODEL_DDPG = "ddpg"

# Initialize VAE model and add it to gym environment.
# VAE does image post processing to latent vector and
# buffers raw image for future optimization.
vae = VAEController()
env.unwrapped.set_vae(vae)

# Run in test mode of trained models exist.
if os.path.exists(PATH_MODEL_DDPG + ".pkl") and \
   os.path.exists(PATH_MODEL_VAE):
    print("Task: test")
    ddpg = DDPG.load(PATH_MODEL_DDPG, env)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        action, _states = ddpg.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()
# Run in training mode.
else:
    print("Task: train")
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            theta=float(0.2) * np.ones(n_actions),
            sigma=float(0.4) * np.ones(n_actions)
            )

    ddpg = DDPG(LnMlpPolicy,
                env,
                verbose=1,
                batch_size=64,
                clip_norm=5e-3,
                gamma=0.9,
                param_noise=None,
                action_noise=action_noise,
                memory_limit=300,
                nb_train_steps=1000,
                )
    ddpg.learn(total_timesteps=1000, vae=vae)
    # Finally save model files.
    ddpg.save(PATH_MODEL_DDPG)
    vae.save(PATH_MODEL_VAE)
