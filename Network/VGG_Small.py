from .BNN import *
import torch.nn as nn


class VGG_Small(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),

            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(128, 128, kernel_size=3, padding=0, stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(128, 256, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),

            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(256, 256, kernel_size=3, padding=0, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
        )
        
        self.conv3 = nn.Sequential(
            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(256, 512, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(512),

            nn.ConstantPad2d(padding=1, value=-1),
            BinaryConv2d(512, 512, kernel_size=3, padding=0, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            
            BinaryLinear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        
        return x