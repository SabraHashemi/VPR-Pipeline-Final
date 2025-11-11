"""NetVLAD placeholder: uses ResNet fallback unless a NetVLAD implementation is available."""
import torch.nn as nn

def build_netvlad(pretrained=False, device='cpu'):
    try:
        from netvlad import build_netvlad as _build
        return _build(pretrained=pretrained, device=device)
    except Exception:
        print('[WARN] NetVLAD not installed. Using ResNet placeholder.')
        import torchvision.models as models
        m=models.resnet50(pretrained=pretrained)
        modules=list(m.children())[:-2]
        backbone=nn.Sequential(*modules)
        pool=nn.AdaptiveAvgPool2d((1,1))
        class P(nn.Module):
            def __init__(self): super().__init__(); self.backbone=backbone; self.pool=pool
            def forward(self,x): f=self.backbone(x); return self.pool(f).view(x.size(0),-1)
        return P()
