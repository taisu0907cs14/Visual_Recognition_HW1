import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_resnet50_baseline(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model