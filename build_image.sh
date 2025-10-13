#! /bin/bash

# Enable BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_VER=12.9.1 \
  --build-arg UBUNTU_VER=22.04 \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t chengcli/kintera:dev \
  -f docker/Dockerfile .
