# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
VAE controller for runtime optimization.
'''

import numpy as np

from .model import ConvVAE


class VAEController:
    def __init__(self, z_size=512, image_size=(80, 160, 3),
                 learning_rate=0.0001, kl_tolerance=0.5,
                 epoch_per_optimization=10, batch_size=64,
                 buffer_size=500):
        # VAE input and output shapes
        self.z_size = z_size
        self.image_size = image_size

        # VAE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Training params
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset()

        self.vae = ConvVAE(z_size=self.z_size,
                           batch_size=self.batch_size,
                           learning_rate=self.learning_rate,
                           kl_tolerance=self.kl_tolerance,
                           is_training=True,
                           reuse=False,
                           gpu_mode=True)

        self.target_vae = ConvVAE(z_size=self.z_size,
                                  batch_size=1,
                                  is_training=False,
                                  reuse=False,
                                  gpu_mode=True)

    def buffer_append(self, arr):
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            self.buffer_pos = 0
            self.buffer_full = True
        self.buffer[self.buffer_pos] = arr

    def buffer_reset(self):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)

    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    def encode(self, arr):
        assert arr.shape == self.image_size
        # Normalize
        arr = arr.astype(np.float)/255.0
        # Reshape
        arr = arr.reshape(1,
                          self.image_size[0],
                          self.image_size[1],
                          self.image_size[2])
        return self.target_vae.encode(arr)

    def decode(self, arr):
        assert arr.shape == (1, self.z_size)
        # Decode
        arr = self.target_vae.decode(arr)
        # Denormalize
        arr = arr * 255.0
        return arr

    def optimize(self):
        ds = self.buffer_get_copy()
        # TODO: may be do buffer reset.
        # self.buffer_reset()

        num_batches = int(np.floor(len(ds)/self.batch_size))

        for epoch in range(self.epoch_per_optimization):
            np.random.shuffle(ds)
            for idx in range(num_batches):
                batch = ds[idx * self.batch_size:(idx + 1) * self.batch_size]
                obs = batch.astype(np.float) / 255.0
                feed = {self.vae.x: obs, }
                (train_loss, r_loss, kl_loss, train_step, _) = self.vae.sess.run([
                    self.vae.loss,
                    self.vae.r_loss,
                    self.vae.kl_loss,
                    self.vae.global_step,
                    self.vae.train_op
                ], feed)
                if ((train_step + 1) % 50 == 0):
                    print("VAE: optimization step",
                          (train_step + 1), train_loss, r_loss, kl_loss)
        self.set_target_params()

    def save(self, path):
        self.target_vae.save_json(path)

    def load(self, path):
        self.target_vae.load_json(path)

    def set_target_params(self):
        params, _, _ = self.vae.get_model_params()
        self.target_vae.set_model_params(params)
