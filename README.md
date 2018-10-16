# learning-to-drive-in-a-day

Code that implement approach similar to described in ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf) paper.

Missing parts:
- Prioritized Experience Replay in DDPG. Right now we randomly sample.
- Params well tuning to drive more smoothly.

# Quick start

NOTE: Assuming Intel Graphics (`/dev/dri`) present.

Download compiled [Donkey Car simulator](https://drive.google.com/open?id=1sK2luxKYV1cpaZLhVwfXrmGU3TRa5C3B) ([source](https://github.com/tawnkramer/sdsandbox/tree/donkey)) into `$HOME/sim` directory.

Run training.
```
docker build -t learning-to-drive-in-a-day .
./run-in-docker.sh
```

Run test with the same command (script run test if there are trained models `ddpg.pkl` and `vae.json`).
```
./run-in-docker.sh
```

# Under the hood

Script does the following:
- Initialize Donkey OpenAI gym environment.
- Initialize VAE controller with random weights.
- If no pretrained models found, run in train mode. Otherwise just load weights from files and run test.
- Initialize DDPG controller.
- Learning function will collect the data by running 5 episodes w/o optimization, then after every episode DDPG and VAE optimization happens.
- After 1000 steps training will be finished and weights params will be saved to files.

# Credits

- [wayve.ai](wayve.ai) for idea and inspiration.
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [stable-baselines](https://github.com/hill-a/stable-baselines) for DDPG implementation.
- [world models experiments](https://github.com/hardmaru/WorldModelsExperiments) for VAE implementation.
