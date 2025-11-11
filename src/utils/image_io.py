from PIL import Image
import os
import csv

def list_images(folder):
    """
    Recursively list image file paths in a folder.
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    all_imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                all_imgs.append(os.path.join(root, f))
    return sorted(all_imgs)

def list_from_csv(csv_path, img_root):
    """
    Reads image filenames from a CSV file and returns full paths.
    Automatically skips headers if present.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    files = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            path = os.path.join(img_root, name)
            if os.path.exists(path):
                files.append(path)
    return sorted(files)

def load_dataset_paths(dataset_path):
    """
    Detect dataset structure automatically.
    Supports both folder-based (database/queries) and CSV-based (csv/*.csv + imgs/).
    Returns (database_paths, query_paths)
    """
    csv_dir = os.path.join(dataset_path, "csv")
    img_dir = os.path.join(dataset_path, "imgs")

    # CSV-based dataset (GSV-XS, Tokyo-XS, SF-XS, etc.)
    if os.path.exists(csv_dir):
        db_csv = os.path.join(csv_dir, "database.csv")
        q_csv = os.path.join(csv_dir, "queries.csv")
        if os.path.exists(db_csv) and os.path.exists(q_csv):
            print(f"[INFO] CSV-based dataset detected: {dataset_path}")
            db_paths = list_from_csv(db_csv, img_dir)
            q_paths = list_from_csv(q_csv, img_dir)
            print(f"  -> Loaded {len(db_paths)} database and {len(q_paths)} query images")
            return db_paths, q_paths

    # Folder-based dataset (default structure)
    db_dir = os.path.join(dataset_path, "database")
    q_dir = os.path.join(dataset_path, "queries")
    if os.path.exists(db_dir) and os.path.exists(q_dir):
        print(f"[INFO] Folder-based dataset detected: {dataset_path}")
        db_paths = list_images(db_dir)
        q_paths = list_images(q_dir)
        print(f"  -> Loaded {len(db_paths)} database and {len(q_paths)} query images")
        return db_paths, q_paths

    # Flat folder fallback
    print(f"[WARN] Unknown dataset structure for {dataset_path}, using flat mode.")
    images = list_images(dataset_path)
    return images, images


def load_rgb(path, resize=None):
    img=Image.open(path).convert('RGB')
    if resize: img=img.resize(resize)
    return img
