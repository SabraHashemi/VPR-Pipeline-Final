from PIL import Image
import os, csv

def list_images(folder, exts=None):
    if exts is None: exts={'.jpg','.jpeg','.png','.bmp'}
    files=[]
    if not os.path.isdir(folder): return files
    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn.lower())[1] in exts: files.append(os.path.join(folder,fn))
    return files

def load_rgb(path, resize=None):
    img=Image.open(path).convert('RGB')
    if resize: img=img.resize(resize)
    return img


def list_from_csv(csv_path, img_root):
    """
    Reads image filenames from a CSV file and returns full paths.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    files = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if not row: continue
            name = row[0].strip()
            path = os.path.join(img_root, name)
            if os.path.exists(path):
                files.append(path)
    return files
