# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
Hijacked donkey_gym wrapper with VAE.

- Use Z vector as observation space.
- Store raw images in VAE buffer.

Problem that DDPG already well implemented in stable-baselines
and VAE integration will require full reimplementation of DDPG
codebase. Instead we hijack VAE into gym environment.
'''

import os

import numpy as np
import gym
from gym import spaces
from donkey_gym.envs.donkey_env import DonkeyEnv
from donkey_gym.envs.donkey_sim import DonkeyUnitySimContoller
from donkey_gym.envs.donkey_proc import DonkeyUnityProcess

class DonkeyVAEEnv(DonkeyEnv):
    def __init__(self, level=0, time_step=0.05, frame_skip=2, z_size=512):
        self.z_size = z_size

        print("starting DonkeyGym env")
        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        try:
            exe_path = os.environ['DONKEY_SIM_PATH']
        except:
            print("Missing DONKEY_SIM_PATH environment var. Using defaults")
            #you must start the executable on your own
            exe_path = "self_start"

        try:
            port = int(os.environ['DONKEY_SIM_PORT'])
        except:
            print("Missing DONKEY_SIM_PORT environment var. Using defaults")
            port = 9090

        try:
            headless = os.environ['DONKEY_SIM_HEADLESS']=='1'
        except:
            print("Missing DONKEY_SIM_HEADLESS environment var. Using defaults")
            headless = False

        self.proc.start(exe_path, headless=headless, port=port)

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(level=level, time_step=time_step, port=port)

        # steering
        # TODO(r7vme): Add throttle
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # z latent vector
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.z_size), dtype=np.float32)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def step(self, action):
        for i in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self._observe()
        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        observation, reward, done, info = self._observe()
        return observation

    def _observe(self):
        observation, reward, done, info = self.viewer.observe()
        # Solves chicken-egg problem as gym calls reset before we call set_vae.
        if not hasattr(self, "vae"):
            return np.zeros(self.z_size), reward, done, info
        # Store image in VAE buffer.
        self.vae.buffer_append(observation)
        return self.vae.encode(observation), reward, done, info

    def set_vae(self, vae):
        self.vae = vae
