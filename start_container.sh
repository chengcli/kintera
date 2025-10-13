#! /usr/bin/bash

# Example host paths:
#   ~/code/your-project       -> /workspace
#   /data/big_dataset         -> /data
docker run -it --rm \
  --gpus all \
  -v ~/Development/kintera:/workspace \
  -v /data/big_dataset:/data:ro \
  -v $HOME/.gitconfig:/home/dev/.gitconfig:ro \
  -v ccache:/ccache \
  -p 8888:8888 \
  cli/kintera:dev
