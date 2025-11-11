#!/usr/bin/env bash
set -e
echo 'Helper script: clones dataset helper repos and prints instructions.'
if [ ! -d "VPR-datasets-downloader" ]; then
  git clone https://github.com/gmberton/VPR-datasets-downloader.git || true
fi
if [ ! -d "MixVPR" ]; then
  git clone https://github.com/amaralibey/MixVPR.git || true
fi
echo 'After cloning, follow the README files of those repos to download raw dataset files.'
echo 'Arrange datasets under datasets/<NAME>/{database,queries}'
