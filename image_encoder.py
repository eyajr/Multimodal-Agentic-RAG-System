import torch
import torch.nn as nn
from labs.diffusion_utilities_context import UnetDown, EmbedFC
from config import VECTOR_SIZE

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, vector_size=VECTOR_SIZE, input_size=32):
        super().__init__()
        # Two UnetDown blocks
        self.down1 = UnetDown(input_channels, base_channels)
        self.down2 = UnetDown(base_channels, base_channels * 2)
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten
        self.flatten = nn.Flatten()
        # EmbedFC to project to vector_size
        self.project = EmbedFC(base_channels * 2, vector_size)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.project(x)
        return x
