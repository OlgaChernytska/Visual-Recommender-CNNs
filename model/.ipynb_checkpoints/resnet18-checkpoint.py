import torch.nn as nn
import torch
from torchvision import models
import numpy as np


class ResNet18(nn.Module):
    '''
    Last fully connected layer changed to ouput 300-dim vector.
    '''
    
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=300, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features
