from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet_v2_s
from torchvision.models import vit_l_16
import torch

class EfficientNet_B0(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier = torch.nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class EfficientNet_V2_S(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = efficientnet_v2_s(pretrained=pretrained)
        self.model.classifier = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ViT_L_16(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = vit_l_16()
        self.model.head = torch.nn.Linear(768, num_classes)
    
    def forward(self, x):
        return self.model(x)