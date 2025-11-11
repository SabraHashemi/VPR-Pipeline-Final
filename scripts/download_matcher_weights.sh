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


#!/usr/bin/env bash
# ================================================
#  Auto Downloader for VPR Matchers & Backbones
#  Supports: NetVLAD, MixVPR, LightGlue, LoFTR
# ================================================

set -e

# Make sure wget or curl exists
if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
  echo "❌ Error: wget or curl is required for downloading files."
  exit 1
fi

# Helper function
download_file() {
  local url=$1
  local output=$2
  if [ -f "$output" ]; then
    echo "✅ Found: $output (skipping)"
  else
    echo "⬇️  Downloading: $output"
    mkdir -p "$(dirname "$output")"
    if command -v wget >/dev/null 2>&1; then
      wget -q --show-progress -O "$output" "$url"
    else
      curl -L -o "$output" "$url"
    fi
  fi
}

echo "==============================="
echo "   VPR Weight Downloader"
echo "==============================="

# --- NetVLAD ---
echo "[NetVLAD]"
download_file \
  "https://github.com/Nanne/pytorch-NetVlad/releases/download/v1/netvlad_checkpoint.pth" \
  "weights/netvlad/netvlad_checkpoint.pth"

# --- MixVPR ---
echo "[MixVPR]"
download_file \
  "https://github.com/amaralibey/MixVPR/releases/download/v1/mixvpr_resnet50.pth" \
  "weights/mixvpr/mixvpr_resnet50.pth"

# --- LightGlue ---
echo "[LightGlue]"
download_file \
  "https://huggingface.co/cvg/LightGlue/resolve/main/weights/superpoint_lightglue.pth" \
  "weights/lightglue/superpoint_lightglue.pth"

# --- LoFTR ---
echo "[LoFTR]"
download_file \
  "https://huggingface.co/zju3dv/LoFTR/resolve/main/weights/outdoor_ds.ckpt" \
  "weights/loftr/outdoor_ds.ckpt"

echo "==============================="
echo "✅ All weights downloaded successfully!"
echo "Saved under: ./weights/"
echo "==============================="
