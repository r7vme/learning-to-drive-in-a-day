# learning-to-drive-in-a-day

Code that implement approach similar to described in ["Learning to Drive in a Day"](https://arxiv.org/pdf/1807.00412.pdf) paper.

# Quick start

NOTE: Assuming Intel Graphics (`/dev/dri`) present.

Download compiled Donkey Car simulator (TBD link) into `$HOME/sim` directory.

Run training.
```
docker build -t learning-to-drive-in-a-day .
./run-in-docker.sh
```
