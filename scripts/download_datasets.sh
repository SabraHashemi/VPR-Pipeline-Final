# #!/usr/bin/env bash
# # ==================================================
# #  Smart Dataset Downloader for VPR Pipeline
# #  Supports: GSV-XS, Tokyo-XS, SF-XS, SVOX
# # ==================================================
# set -e

# BASE_DIR="datasets"
# mkdir -p "$BASE_DIR"

# echo "============================================="
# echo "📦  VPR Dataset Auto Downloader (Fixed Unzip)"
# echo "Target folder: $BASE_DIR/"
# echo "============================================="

# download_zip() {
#   local name=$1
#   local url=$2
#   local target="$BASE_DIR/$name"
#   local zip_path="${BASE_DIR}/${name}.zip"
#   local tmp_dir="${BASE_DIR}/tmp_${name}"

#   if [ -d "$target" ]; then
#     echo "✅ Dataset '$name' already exists at $target — skipping."
#     return
#   fi

#   echo "⬇️  Downloading dataset: $name"
#   wget -q --show-progress -O "$zip_path" "$url"

#   echo "📂 Extracting..."
#   mkdir -p "$tmp_dir"
#   unzip -qq "$zip_path" -d "$tmp_dir"
#   rm -f "$zip_path"

#   # Move inner folder contents up
#   subdir=$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
#   if [ -n "$subdir" ]; then
#     mkdir -p "$target"
#     mv "$subdir"/* "$target"/
#   else
#     # fallback if zip had no subfolder
#     mv "$tmp_dir" "$target"
#   fi
#   rm -rf "$tmp_dir"

#   echo "✅ Extracted: $target"
#   echo "---------------------------------------------"
# }

# download_zip "GSV-XS" "https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf/gsv_xs.zip"
# #download_zip "Tokyo-XS" "https://github.com/amaralibey/MixVPR/releases/download/data/Tokyo_xs_sample.zip"
# #download_zip "SF-XS" "https://github.com/amaralibey/MixVPR/releases/download/data/SF_xs_sample.zip"
# #download_zip "SVOX" "https://github.com/amaralibey/MixVPR/releases/download/data/SVOX_sample.zip"

# echo "============================================="
# echo "✅ All datasets extracted correctly!"
# echo "Datasets ready under ./datasets/"
# echo "============================================="

#!/usr/bin/env bash
# ==================================================
#  Smart Dataset Downloader (Google Drive folders)
# ==================================================
set -e

BASE_DIR="datasets"
mkdir -p "$BASE_DIR"

echo "============================================="
echo "📦  VPR Dataset Auto Downloader (Drive folder)"
echo "Target folder: $BASE_DIR/"
echo "============================================="

# --- Ensure gdown is installed ---
if ! command -v gdown >/dev/null 2>&1; then
  echo "📥 Installing gdown..."
  pip install gdown
fi

# --- Function: Download from Google Drive folder ---
download_from_drive_folder() {
  local name=$1
  local folder_id=$2
  local target="$BASE_DIR/$name"

  if [ -d "$target" ]; then
    echo "✅ Dataset '$name' already exists at $target — skipping."
    return
  fi

  echo "⬇️  Downloading Google Drive folder for $name ..."
  mkdir -p "$target"
  gdown --folder "https://drive.google.com/drive/folders/${folder_id}" -O "$target"

  echo "---------------------------------------------"
  echo "✅ Downloaded files for $name → $target"
  echo "---------------------------------------------"

  # --- Optional: auto-unzip if a zip file exists inside ---
  zip_file=$(find "$target" -type f -name "*.zip" | head -n 1)
  if [ -n "$zip_file" ]; then
    echo "📂 Extracting ZIP inside folder..."
    unzip -qq "$zip_file" -d "$target"
    rm -f "$zip_file"
    echo "✅ Extracted and cleaned up ZIP."
  fi
}

# --- Do
