import torch.nn as nn
import torch.nn.functional as F
import torch
from unet_utils import *

# this is the previous a2s net
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # first layer: 15 -> 32
        self.fc1 = nn.Linear(in_features=15,out_features=32)
        # second layer: 32 -> 32
        self.fc2 = nn.Linear(in_features=32,out_features=32)
        # third layer: 32 -> 64
        self.fc3 = nn.Linear(in_features=32,out_features=64)
        # fifth layer: 64 -> 32
        self.fc4 = nn.Linear(in_features=64,out_features=32)
        # sixth layer: 32 -> 16
        self.fc5 = nn.Linear(in_features=32,out_features=16)
        # output layer
        self.out = nn.Linear(in_features=16,out_features=3)

        self.bn1 = nn.LayerNorm(32)
        self.bn2 = nn.LayerNorm(32)
        self.bn3 = nn.LayerNorm(64)
        self.bn4 = nn.LayerNorm(32)
        self.bn5 = nn.LayerNorm(16)

    def forward(self,t):
        
        t = self.fc1(t)
        t = F.elu(t)
        t = self.bn1(t)
        
        t = self.fc2(t)
        t = F.elu(t)
        t = self.bn2(t)

        t = self.fc3(t)
        t = F.elu(t)
        t = self.bn3(t)

        t = self.fc4(t)
        t = F.elu(t)
        t = self.bn4(t)

        t = self.fc5(t)
        t = F.elu(t)
        t = self.bn5(t)

        t = self.out(t)
        return t

# decoder has a reverse sturcture of net
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # first layer: 3 -> 16
        self.fc1 = nn.Linear(in_features=3,out_features=16)
        # second layer: 16 -> 32
        self.fc2 = nn.Linear(in_features=16,out_features=32)
        # third layer: 32 -> 64
        self.fc3 = nn.Linear(in_features=32,out_features=64)
        # forth layer: 64 -> 32
        self.fc4 = nn.Linear(in_features=64,out_features=32)
        # fifth layer: 32 -> 32
        self.fc5 = nn.Linear(in_features=32,out_features=32)
        # out layer: 32 -> 3
        self.out = nn.Linear(in_features=32,out_features=3)

        self.bn1 = nn.LayerNorm(16)
        self.bn2 = nn.LayerNorm(32)
        self.bn3 = nn.LayerNorm(64)
        self.bn4 = nn.LayerNorm(32)
        self.bn5 = nn.LayerNorm(32)

    def forward(self,t):
        
        t = self.fc1(t)
        t = F.elu(t)
        t = self.bn1(t)
        
        t = self.fc2(t)
        t = F.elu(t)
        t = self.bn2(t)

        t = self.fc3(t)
        t = F.elu(t)
        t = self.bn3(t)

        t = self.fc4(t)
        t = F.elu(t)
        t = self.bn4(t)

        t = self.fc5(t)
        t = F.elu(t)
        t = self.bn5(t)

        t = self.out(t)
        return t

# unet part
class UNet(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(UNet,self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch,filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        print(f"UNET input is {in_ch} channels")

    def forward(self,x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out





