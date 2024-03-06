from torchvision.models import efficientnet_b0
import torch

class EfficientNet_B0(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier = torch.nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.model(x)