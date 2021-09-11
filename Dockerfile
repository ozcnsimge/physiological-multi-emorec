FROM tensorflow/tensorflow:1.15.2-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

## Install requirements
RUN apt-get update && apt-get install -y git nano protobuf-compiler python-pil python-lxml python-tk vim unoconv libsndfile1

COPY utils/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

## Non-root operation
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME