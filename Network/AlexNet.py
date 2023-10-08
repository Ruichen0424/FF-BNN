from .BNN import *
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, padding=2, stride=4, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            
            nn.ConstantPad2d(padding=2, value=-1),
            BinaryConv2d(96, 256, kernel_size=5, padding=0, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(256, 384, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(384),
            
            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(384, 384, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(384),
        )
        
        
        self.conv3 = nn.Sequential(
            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(384, 256, kernel_size=3, padding=0, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            
            BinaryLinear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        
        return x