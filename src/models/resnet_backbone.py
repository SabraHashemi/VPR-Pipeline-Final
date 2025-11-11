import torch.nn as nn, torchvision.models as models
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True, device='cpu'):
        super().__init__(); self.device=device
        m=models.resnet50(pretrained=pretrained)
        self.backbone=nn.Sequential(*list(m.children())[:-2]); self.pool=nn.AdaptiveAvgPool2d((1,1)); self.to(device)
    def forward(self,x): f=self.backbone(x); return self.pool(f).view(x.size(0),-1)
def build_resnet50(pretrained=True, device='cpu'): return ResNet50Backbone(pretrained=pretrained, device=device)
