#!/bin/bash

IMG_NAME="dev-env-simge"

docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t $IMG_NAME