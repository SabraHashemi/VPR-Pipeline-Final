#!/usr/bin/env bash
# ==================================================
#  Universal Dataset Downloader for VPR Pipeline
# ==================================================
set -e

BASE_DIR="datasets"
mkdir -p "$BASE_DIR"

echo "============================================="
echo "📦  VPR Dataset Downloader (Final Version)"
echo "Target folder: $BASE_DIR/"
echo "============================================="

# --- make sure gdown exists ---
if ! command -v gdown >/dev/null 2>&1; then
  echo "📥 Installing gdown..."
  pip install -q gdown
fi

# --- define download function ---
download_from_drive_file() {
  local name=$1        # dataset name, e.g. GSV-XS
  local file_id=$2     # Google Drive file ID
  local target="$BASE_DIR/$name"
  local zip_path="${BASE_DIR}/${name}.zip"

  echo "⬇️  Downloading ${name}.zip from Google Drive..."
  mkdir -p "$BASE_DIR"
  
  # clean old mess
  rm -rf "$target"
  rm -f "$zip_path"

  # ✅ Download the zip file directly to datasets/
  gdown "https://drive.google.com/uc?id=${file_id}" -O "$zip_path"

  echo "📂 Extracting..."
  mkdir -p "$target"
  unzip -qq "$zip_path" -d "$target"
  rm -f "$zip_path"

  # ✅ flatten if extra nested folder exists
  inner_dir=$(find "$target" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ -n "$inner_dir" ]; then
    echo "📁 Fixing nested structure..."
    mv "$inner_dir"/* "$target"/ 2>/dev/null || true
    rm -rf "$inner_dir"
  fi

  echo "✅ Dataset ready → $target"
  echo "---------------------------------------------"
}

# --- Download datasets ---
#download_from_drive_file "gsv_xs" "1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A"
download_from_drive_file "tokyo_xs" "1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A"
# download_from_drive_file "SF-XS" "YOUR_FILE_ID_HERE"
# download_from_drive_file "SVOX" "YOUR_FILE_ID_HERE"

echo "============================================="
echo "✅ All datasets are ready under ./datasets/"
echo "============================================="
