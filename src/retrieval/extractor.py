import torch, numpy as np
from torchvision import transforms
from PIL import Image

def default_transform(size=480):
    return transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def extract_descriptors(model, image_paths, batch_size=8, device='cpu', size=480):
    model.eval(); tf=default_transform(size); descs=[]; paths=[]
    with torch.no_grad():
        for i in range(0,len(image_paths),batch_size):
            batch=image_paths[i:i+batch_size]; imgs=[]
            for p in batch: imgs.append(tf(Image.open(p).convert('RGB')))
            x=torch.stack(imgs).to(device); feats=model(x)
            if hasattr(feats,'cpu'): feats=feats.cpu().numpy()
            norms=(feats**2).sum(axis=1,keepdims=True)**0.5; feats=feats/(norms+1e-10)
            descs.append(feats); paths.extend(batch)
    return np.vstack(descs) if descs else np.zeros((0,2048)), paths
