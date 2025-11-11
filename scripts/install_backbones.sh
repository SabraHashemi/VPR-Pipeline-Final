#!/usr/bin/env bash
# ==================================================
#  Install Backbones & Matchers for VPR Pipeline
# ==================================================
set -e

echo "======================================="
echo "⚙️  Installing VPR backbones and matchers..."
echo "======================================="

# Ensure pip is updated
pip install --upgrade pip setuptools wheel

# --- NetVLAD ---
echo "[1/4] Installing NetVLAD..."
pip install git+https://github.com/Nanne/pytorch-NetVlad.git || {
  echo "⚠️  NetVLAD installation failed, continuing..."
}

# --- MixVPR ---
echo "[2/4] Installing MixVPR..."
pip install git+https://github.com/amaralibey/MixVPR.git || {
  echo "⚠️  MixVPR installation failed, continuing..."
}

# --- LightGlue ---
echo "[3/4] Installing LightGlue..."
pip install git+https://github.com/cvg/LightGlue.git || {
  echo "⚠️  LightGlue installation failed, continuing..."
}

# --- LoFTR ---
echo "[4/4] Installing LoFTR..."
pip install git+https://github.com/zju3dv/LoFTR.git || {
  echo "⚠️  LoFTR installation failed, continuing..."
}

echo "======================================="
echo "✅ All backbones and matchers installed!"
echo "======================================="
