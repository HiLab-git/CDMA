import torchvision
from torch import nn
import torchvision.models as models
import torch


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # resnet50
        net = models.resnet50(pretrained=True)
        self.model = net
        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)
        # self.fc 

    def forward(self, x):
        x = self.net.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        return x

class ResNet50_fc(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # resnet50
        net = models.resnet50(pretrained=True)
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.fc 
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        return x