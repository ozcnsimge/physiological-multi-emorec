#!/bin/bash

#path to repo
DIR_PATH=$(pwd)

#image name
IMG_NAME="dev-env-simge"

docker run -it --gpus device=0 --rm -v $DIR_PATH:/repo \
                                -v "$DIR_PATH/clean_data":/data \
                                --workdir="/repo" \
                                --name "$IMG_NAME" \
                                "$IMG_NAME" bash
