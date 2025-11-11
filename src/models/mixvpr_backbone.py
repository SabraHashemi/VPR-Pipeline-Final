"""MixVPR placeholder: uses ResNet fallback unless MixVPR is installed."""
import torch.nn as nn

def build_mixvpr(pretrained=False, device='cpu'):
    try:
        from mixvpr import build_mixvpr as _build
        return _build(pretrained=pretrained, device=device)
    except Exception:
        print('[WARN] MixVPR not installed. Using ResNet placeholder.')
        import torchvision.models as models
        m=models.resnet50(pretrained=pretrained)
        modules=list(m.children())[:-2]
        backbone=nn.Sequential(*modules)
        pool=nn.AdaptiveAvgPool2d((1,1))
        class P(nn.Module):
            def __init__(self): super().__init__(); self.backbone=backbone; self.pool=pool
            def forward(self,x): f=self.backbone(x); return self.pool(f).view(x.size(0),-1)
        return P()
