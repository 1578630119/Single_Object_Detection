import torch
import torch.nn.functional as F

import math

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32, 3,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1,stride=2)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1,stride=2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(128, 512, 3, padding=1, stride=2)
        self.avgpool = nn.AvgPool2d((2,2))
        self.fc1 = nn.Linear(512*10*13, 512)
        self.fc2 = nn.Linear(512, 4)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):                   #x (3,h,w)
            x = F.leaky_relu(self.conv1(x)) #32,h/2,w/2
            x = F.leaky_relu(self.conv2(x)) #32,h/4,w/4            #32,h/4,w/4
            x = F.leaky_relu(self.conv3(x)) #128,h/8,w/8
            x = F.leaky_relu(self.conv4(x)) #128,h/16,w/16             #128,h/16,w/16
            x = F.leaky_relu(self.conv5(x)) #512,h/32,w/32

            x = x.view(-1, 512*10*13)
            x = F.leaky_relu(self.fc1(x))
            x=self.fc2(x)
            return x


