#!/bin/bash

url_root="https://dl.fbaipublicfiles.com/adaptive-span"

mkdir -p checkpoints
cd checkpoints

wget "${url_root}/enwik8.pt"
wget "${url_root}/text8.pt"
wget "${url_root}/enwik8_large.pt"
wget "${url_root}/text8_large.pt"
