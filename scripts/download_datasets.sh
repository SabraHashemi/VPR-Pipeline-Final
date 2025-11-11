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
download_from_drive_file() {
  local name=$1        # e.g. GSV-XS
  local file_id=$2     # e.g. 1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A
  local target="$BASE_DIR/$name"   # datasets/GSV-XS
  local zip_path="${BASE_DIR}/${name}.zip"  # datasets/GSV-XS.zip

  echo "⬇️  Downloading ${name}.zip from Google Drive..."
  mkdir -p "$BASE_DIR"

  # ✅ 
  gdown "https://drive.google.com/uc?id=${file_id}" -O "$zip_path"

  echo "📂 Extracting..."
  mkdir -p "$target"
  unzip -qq "$zip_path" -d "$target"
  rm -f "$zip_path"

  # ✅ 
  inner_dir=$(find "$target" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ -n "$inner_dir" ]; then
    echo "📁 Fixing nested structure..."
    mv "$inner_dir"/* "$target"/
    rm -rf "$inner_dir"
  fi

  echo "✅ Extracted cleanly → $target"
  echo "---------------------------------------------"
}



# --- Download actual datasets ---
# (Use your real Google Drive folder IDs)
download_from_drive_folder "gsv_xs" "1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf"
# download_from_drive_folder "Tokyo-XS" "YOUR_FOLDER_ID_HERE"
# download_from_drive_folder "SF-XS" "YOUR_FOLDER_ID_HERE"
# download_from_drive_folder "SVOX" "YOUR_FOLDER_ID_HERE"

echo "============================================="
echo "✅ All datasets ready under ./datasets/"
echo "============================================="