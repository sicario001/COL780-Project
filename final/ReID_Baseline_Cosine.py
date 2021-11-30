# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/sicario001/COL780-Project/blob/main/final/ReID_Baseline_Cosine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
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
import cv2
# %%
from torch.nn import Conv2d,Linear, Module, Sequential,BatchNorm2d,MaxPool2d
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

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
class ReidModel1(Module):   
    def __init__(self, numClasses = 2, inference = False):
        super(ReidModel1, self).__init__()
        self.inference = inference
        self.numClasses = numClasses
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

        self.dense_10 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128*16*6,out_features=128),
            nn.BatchNorm1d(128),
            nn.ELU()
        )
        
        self.cosine_weights = nn.Parameter(torch.randn(128, numClasses))
        self.k = nn.Parameter(torch.tensor(1.))

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
        out = self.dense_10(out)
        out = nn.functional.normalize(out,p=2,dim=1)
        if (not self.inference):
          weights_norm = nn.functional.normalize(self.cosine_weights,p=2,dim=0)
          # print(weights_norm[:,0]@weights_norm[:,0])
          out = self.k * (out @ weights_norm)
        return out


# %%
class ReidModelResnet50(Module):
  def __init__(self, numClasses = 2, inference = False):
    super(ReidModelResnet50, self).__init__()
    self.inference = inference
    self.numClasses = numClasses
    # load pretrained resnet50
    base_model = models.resnet50(pretrained = True, progress= False)
    base_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
    # freeze weights
    num_layers = len(list(base_model.children()))
    ct = 0
    for child in base_model.children():
      ct += 1
      if ct < (num_layers-2):
          for param in child.parameters():
              param.requires_grad = False
    self.base_model = base_model
    self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048,out_features=128),
            nn.BatchNorm1d(128),
            nn.ELU()
        )
    self.cosine_layer = nn.utils.weight_norm(nn.Linear(in_features=128,out_features=numClasses,bias=False))
  def forward(self, x):
    out = self.base_model(x)
    # out = torch.squeeze(out, dim = 3)
    # out = torch.squeeze(out, dim = 2)
    out = self.final_layer(out)
    out = nn.functional.normalize(out,p=2,dim=1)
    if (not self.inference):
      out = self.cosine_layer(out)
    return out