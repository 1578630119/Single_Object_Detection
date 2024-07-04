import torch
import torch.nn.functional as F
import torchvision
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

class Net_res(nn.Module):
    def __init__(self):
        super(Net_res, self).__init__()
        resnet18=torchvision.models.resnet18(pretrained=True)
        # ----------------------------------------------------------------------------#
        #   获取特征提取部分，从conv1到model.layer3，最终获得一个h/16,w/16,256的特征层
        # ----------------------------------------------------------------------------#
        features = list([resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool, resnet18.layer1, resnet18.layer2])
        self.features = nn.Sequential(*features)
        self.layer3=resnet18.layer3
        self.fc1 = nn.Linear(26*20*256, 1024)
        self.fc2 = nn.Linear(1024, 4)


    def forward(self, x):
        x=self.features(x)
        x=self.layer3(x)
        x = x.view(-1, 26*20*256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DIou_Loss(nn.Module):
    def __init__(self):
        super(DIou_Loss, self).__init__()

    def box_iou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch,[x1,y1,x2,y2]) x1,y1 左上角坐标，x2，y2右下角坐标
        b2: tensor, shape=(batch, [x1,y1,x2,y2])
        返回为：
        -------
        iou: tensor, shape=(batch, 1)
        """
        b1_wh = b1[:, 2:4] - b1[:, :2]
        b2_wh = b2[:, 2:4] - b2[:, :2]
        inter_x1 = torch.max(b1[:, 0], b2[:, 0])
        inter_y1 = torch.max(b1[:, 1], b2[:, 1])
        inter_x2 = torch.min(b1[:, 2], b2[:, 2])
        inter_y2 = torch.min(b1[:, 3], b2[:, 3])


        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_area =  (torch.clamp(inter_x2 - inter_x1, min=0)+1) * (torch.clamp(inter_y2 - inter_y1, min=0)+1)
        b1_area = (b1_wh[:, 0]+1) * (b1_wh[:, 1]+1)
        b2_area = (b2_wh[:, 0]+1) * (b2_wh[:, 1]+1)
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # DISTANCE
        C_x1 = torch.min(b1[...,0], b2[...,0])
        C_y1 = torch.min(b1[...,1], b2[...,1])
        C_x2 = torch.max(b1[...,2], b2[...,2])
        C_y2 = torch.max(b1[...,3], b2[...,3])

        center_x1 = (b1[...,0] + b1[...,2]) / 2
        center_y1 = (b1[...,1] + b1[...,3]) / 2
        center_x2 = (b2[...,0] + b2[...,2]) / 2
        center_y2 = (b2[...,1] + b2[...,3]) / 2

        center_distance = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        c_distance = (C_x2 - C_x1) ** 2 + (C_y2 - C_y1) ** 2

        DIOU = iou - center_distance / c_distance


        return DIOU

    def forward(self, input, targets=None):
        iou = self.box_iou(input, targets)  # 计算交互比
        loss = torch.mean((1 - iou))
        return loss
