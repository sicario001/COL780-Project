# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image


# %%
import os
import sys

# %%
from utils import get_id


# %%
import cv2

# %% [markdown]
# %%
from torch.nn import Conv2d,Linear, Module, Sequential,BatchNorm2d,MaxPool2d
import torch.nn.functional as F


# %%
class ResidualBlock(Module):
  def __init__(self,in_channels,increase_dims=False,is_first=False):
        super(ResidualBlock, self).__init__()
        self.increase_dims = increase_dims
        self.is_first = is_first
        self.bn_0 = None
        self.conv_up = None
        out_channels = in_channels
        stride = 1
        if increase_dims:
          out_channels*=2
          stride*=2
          # self.conv_up = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2,padding="same")
          self.conv_up = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)
        if not is_first:
          self.bn_0 = nn.Sequential(
              nn.BatchNorm2d(in_channels),
              nn.ELU()
          )
        if increase_dims:
          self.conv_1 = nn.Sequential(
              # nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding="same"),
              nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride),
              nn.BatchNorm2d(out_channels),
              nn.ELU(),
              nn.Dropout(p=0.4)
          )
        else:
          self.conv_1 = nn.Sequential(
              nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding="same"),
              # nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride),
              nn.BatchNorm2d(out_channels),
              nn.ELU(),
              nn.Dropout(p=0.4)
          )
        self.conv_2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding="same")
  def forward(self,x):
      if self.bn_0:
        y = self.bn_0(x)
      else:
        y = x
      residual = x
      if self.increase_dims:
        y = F.pad(y, (0, 1, 0, 1))
      out = self.conv_1(y)
      out = self.conv_2(out)
      if self.conv_up:
        residual = self.conv_up(residual)
      return out+residual


# %%
class BigNet(Module):   
    def __init__(self):
        super(BigNet, self).__init__()
        self.conv_1 = nn.Sequential(
          nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding="same"),
          nn.BatchNorm2d(32),
          nn.ELU()
        )
        self.conv_2 = nn.Sequential(
          nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding="same"),
          nn.BatchNorm2d(32),
          nn.ELU()
        )
        # self.pool_3 = nn.MaxPool2d(kernel_size=3,stride=2,padding="same")
        self.pool_3 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.residual_4 = ResidualBlock(in_channels=32,is_first=True)
        self.residual_5 = ResidualBlock(in_channels=32)
        self.residual_6 = ResidualBlock(in_channels=32,increase_dims=True)
        self.residual_7 = ResidualBlock(in_channels=64)
        self.residual_8 = ResidualBlock(in_channels=64,increase_dims=True)
        self.residual_9 = ResidualBlock(in_channels=128)

        

    # Defining the forward pass    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = F.pad(out, (0, 1, 0, 1))
        out = self.pool_3(out)
        out = self.residual_4(out)
        out = self.residual_5(out)
        out = self.residual_6(out)
        out = self.residual_7(out)
        out = self.residual_8(out)
        out = self.residual_9(out)
        # out = self.dense_10(out)
        # out = nn.functional.normalize(out,p=2,dim=1)
        # out = self.cosine_layer(out)
        return out


# %%
class SliceNet(Module):
  def __init__(self, numClasses = 2, inference = False):
    super(SliceNet, self).__init__()
    self.inference = inference
    self.numClasses = numClasses
    self.bignet = BigNet()
    self.dense_10 = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features=128*4*6,out_features=32),
        nn.BatchNorm1d(32),
        nn.ELU()
    )
    self.cosine_layer = nn.utils.weight_norm(nn.Linear(in_features=128,out_features=numClasses,bias=False))

  def forward(self, x):
    out = self.bignet(x)
    # print(out.shape)
    x_slices = torch.split(out,split_size_or_sections=4,dim=2)
    # print(x_slices[0].shape)
    out_slices = []
    for x_slice in x_slices:
      out = self.dense_10(x_slice)
      out_slices.append(out)
    # print(out_slices[0].shape)
    out = torch.cat(out_slices,dim=1)
    out = nn.functional.normalize(out,p=2,dim=1)
    # print(out.shape)
    # assert False
    if (not self.inference):
      out = self.cosine_layer(out)
    return out

