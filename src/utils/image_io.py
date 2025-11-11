from PIL import Image
import os

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
