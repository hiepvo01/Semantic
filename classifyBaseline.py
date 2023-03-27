import os
import numpy as np
from matplotlib import pyplot
from torchvision import datasets
from torchvision import transforms
import torch
from torch import nn

class Trans(torch.nn.Module):
    def __init__(self, img_shape, hid_dim):
        super().__init__()
        self.img_shape = img_shape
        self.hid_dim = hid_dim
        
        self.trans = torch.nn.Sequential(
            nn.Linear(self.img_shape, self.hid_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 4, self.hid_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        transmitted = self.trans(x)
        return transmitted
    
class Receiver(torch.nn.Module):
    def __init__(self, img_shape, hid_dim):
        super().__init__()
        self.img_shape = img_shape
        self.hid_dim = hid_dim
        
        self.rec = torch.nn.Sequential(
            nn.Linear(self.img_shape, self.hid_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 4, self.hid_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 2, self.hid_dim),
        )
        
    def forward(self, x):
        received = self.rec(x)
        return received
    


