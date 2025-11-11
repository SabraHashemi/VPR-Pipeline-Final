#!/usr/bin/env bash
# set -e
# echo 'Helper script: clones dataset helper repos and prints instructions.'
# if [ ! -d "VPR-datasets-downloader" ]; then
#   git clone https://github.com/gmberton/VPR-datasets-downloader.git || true
# fi
# if [ ! -d "MixVPR" ]; then
#   git clone https://github.com/amaralibey/MixVPR.git || true
# fi
# echo 'After cloning, follow the README files of those repos to download raw dataset files.'
# echo 'Arrange datasets under datasets/<NAME>/{database,queries}'

#!/usr/bin/env bash
# ==================================================
#  Automatic Dataset Downloader for VPR Pipeline
#  Supports: GSV-XS, Tokyo-XS, SF-XS, SVOX
# ==================================================
set -e

BASE_DIR="datasets"
mkdir -p "$BASE_DIR"

echo "============================================="
echo "📦  VPR Dataset Auto Downloader"
echo "Target folder: $BASE_DIR/"
echo "============================================="

# Helper function
download_zip() {
  local name=$1
  local url=$2
  local target="$BASE_DIR/$name"
  local zip_path="${target}.zip"

  if [ -d "$target" ]; then
    echo "✅ Dataset '$name' already exists at $target — skipping."
    return
  fi

  echo "⬇️  Downloading dataset: $name"
  mkdir -p "$target"
  wget -q --show-progress -O "$zip_path" "$url"

  echo "📂 Extracting..."
  unzip -qq "$zip_path" -d "$target"
  rm -f "$zip_path"

  # Normalize structure
  if [ -d "$target/database" ] && [ -d "$target/queries" ]; then
    echo "✅ Structure OK for $name"
  else
    echo "⚙️  Trying to auto-fix folder layout..."
    subdir=$(find "$target" -maxdepth 1 -type d ! -path "$target" | head -n 1)
    if [ -n "$subdir" ]; then
      mv "$subdir"/* "$target"/ 2>/dev/null || true
      rm -rf "$subdir"
    fi
    mkdir -p "$target/database" "$target/queries"
    find "$target" -maxdepth 1 -type f -name "*db*" -exec mv {} "$target/database/" \; || true
    find "$target" -maxdepth 1 -type f -name "*query*" -exec mv {} "$target/queries/" \; || true
  fi

  echo "✅ Finished preparing: $name"
  echo "---------------------------------------------"
}

# --- Example small/medium datasets ---
# (These are open/public sample datasets. Replace URLs if needed.)
download_zip "GSV-XS" "https://github.com/amaralibey/MixVPR/releases/download/data/GSV_xs_sample.zip"
download_zip "Tokyo-XS" "https://github.com/amaralibey/MixVPR/releases/download/data/Tokyo_xs_sample.zip"
download_zip "SF-XS" "https://github.com/amaralibey/MixVPR/releases/download/data/SF_xs_sample.zip"
download_zip "SVOX" "https://github.com/amaralibey/MixVPR/releases/download/data/SVOX_sample.zip"

echo "============================================="
echo "✅ All datasets are ready under ./datasets/"
echo "============================================="
