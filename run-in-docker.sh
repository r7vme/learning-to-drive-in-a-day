#!/bin/bash

# Usage: ./run-in-docker.sh
#
# NOTE: Assumed that laptop has Intel Graphics. Remove "/dev/dri" mount to use virtual grpahics (slow).

docker run --net host \
    --rm \
    -ti \
    -e DISPLAY \
    -e DONKEY_SIM_HEADLESS=0 \
    -e DONKEY_SIM_PATH=/sim/donkey_sim.x86_64 \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $HOME/sim:/sim \
    -v $(pwd):/code \
    -w /code \
    --device=/dev/dri:/dev/dri \
        learning-to-drive-in-a-day ./run.py
