#!/usr/bin/env bash
set -e

echo "Manual instructions to obtain pretrained weights for NetVLAD, MixVPR, LightGlue, and LoFTR."

echo "NetVLAD:"
echo "  - Example repo: https://github.com/Nanne/pytorch-NetVlad"
echo "  - Download the checkpoint (e.g. netvlad_checkpoint.pth) and place under: weights/netvlad/"

echo "MixVPR:"
echo "  - Repo: https://github.com/amaralibey/MixVPR"
echo "  - Follow README to obtain weights and place under: weights/mixvpr/"

echo "LightGlue / LoFTR:"
echo "  - See respective repos or HuggingFace model pages for downloads; place weights under weights/lightglue/ or weights/loftr/"

echo "This script intentionally prints instructions rather than auto-downloading large binary files."
