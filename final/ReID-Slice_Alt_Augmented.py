# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/sicario001/COL780-Project/blob/main/final/ReID-Slice_Alt_Augmented.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
get_ipython().system('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1qgRXM0ZiX5_L0V-e1cr_clc2uHDFVcyz\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1qgRXM0ZiX5_L0V-e1cr_clc2uHDFVcyz" -O starter_code.zip && rm -rf /tmp/cookies.txt')


# %%
get_ipython().system('unzip -o starter_code')


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
starter_path = os.path.abspath(os.getcwd())+'/reid-col780-master'
sys.path.insert(0, starter_path)


# %%
from utils import get_id


# %%
import cv2
imgSample = cv2.imread(starter_path+'/data/train/001/01_1.png')
plt.imshow(imgSample)
imgDim = imgSample.shape
print(imgDim)

# %% [markdown]
# ### Load Dataset

# %%
def getCameraID(fileName):
  id = int(fileName.split('_')[0])
  return id


# %%
augment_dataset = True
trainImages = os.listdir(starter_path+'/data/train')
numClasses = len(trainImages)
trainImages = [[label, getCameraID(y), trainImages[label]+'/'+y] for label in range(len(trainImages)) for y in os.listdir(starter_path+'/data/train/'+trainImages[label]) ]
trainImagesFlipped = []
for i in range(len(trainImages)):
  img_path = os.path.join(starter_path+'/data/train', trainImages[i][2])
  image = read_image(img_path)
  # dataset augmentation
  image_flipped = torch.flip(image, (2,))
  trainImages[i][2] = image
  if (augment_dataset):
    trainImagesFlipped.append(trainImages[i][:])
    trainImagesFlipped[i][2] = image_flipped
trainImages.extend(trainImagesFlipped)
print(len(trainImages))


# %%
transform_train_list = [
        transforms.ToPILImage(),
        transforms.Resize((128,48)),
        transforms.ToTensor(),
    ]
transform_train = transforms.Compose(transform_train_list)
target_transform = transforms.Lambda(lambda y: torch.zeros(numClasses, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

# %% [markdown]
# ### Normalization

# %%
imgsCam = torch.stack([transform_train(x[2]) for x in trainImages], dim=3)
meanCam = imgsCam.view(3, -1).mean(dim=1)
stdCam = imgsCam.view(3, -1).std(dim=1)


# %%
tranformNormCam = transforms.Normalize(meanCam, stdCam)
for x in trainImages:
  x[2] = tranformNormCam(transform_train(x[2]))

# %% [markdown]
# ### Custom Dataset

# %%
class CustomDataset(Dataset):
  def __init__(self, images, transform=None, target_transform=None):
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

  def __len__(self):
        return len(self.images)

  def __getitem__(self, idx):
        label = self.images[idx][0]
        image = self.images[idx][2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# %%
dataset = CustomDataset(trainImages, target_transform=target_transform)

# %% [markdown]
# ### Model

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

# %% [markdown]
# ### Training

# %%
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from sklearn.model_selection import KFold


# %%
k_folds = 5
num_epochs = 200
batch_size = 16
torch.manual_seed(42)


# %%
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
      # print(f'Reset trainable parameters of layer = {layer}')
      layer.reset_parameters()

# %% [markdown]
# Use GPU if available

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def reset_net():
  global net
  net = SliceNet(numClasses = 62, inference = False)
  net.to(device)
reset_net()


# %%
def train_epoch(trainDataset,criterion,optimizer):

    num_minibatches = len(trainDataset)//batch_size
    running_loss = 0.0
    for i, data in enumerate(trainDataset, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % (num_minibatches//5) == (num_minibatches//5-1):    # print every 100 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / (num_minibatches//5)))
        #     running_loss = 0.0
    return running_loss

# %% [markdown]
# #### Test Set

# %%
def test(testDataset,criterion):
    correct = 0
    loss = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testDataset:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            y_pred = net(images)

            # cross entropy loss
            loss += criterion(y_pred,labels).item()

            # get labels
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(y_pred_softmax, dim = 1) 
            _, labels = torch.max(labels, dim = 1) 

            # get accuracy
            correct += (predicted == labels).sum().item()

            # total items
            total += labels.size(0)

    return loss,100 * correct / total

# %% [markdown]
# #### KFold

# %%
from sklearn.model_selection import KFold


# %%
kfold = KFold(n_splits=k_folds,shuffle=True,random_state=42)


# %%
# results = {}
# # K-fold Cross Validation model evaluation
# for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
  
#   # Print
#   print(f'FOLD {fold}')
#   print('--------------------------------')
  
#   # Sample elements randomly from a given list of ids, no replacement.
#   train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#   test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
  
#   # Define data loaders for training and testing data in this fold
#   trainloader = torch.utils.data.DataLoader(
#                     dataset, 
#                     batch_size=batch_size, sampler=train_subsampler)
#   testloader = torch.utils.data.DataLoader(
#                     dataset,
#                     batch_size=batch_size, sampler=test_subsampler)
  
#   # reset
#   reset_net()
#   criterion = nn.CrossEntropyLoss()
#   optimizer = Adam(net.parameters())
#   # train
#   for epoch in range(num_epochs):
#     loss = train_epoch(trainloader,criterion,optimizer)
#     val_loss,val_acc = test(testloader,criterion)
#     print(f'Epoch\t{epoch+1}\tloss:{loss:.4f}\tval_loss:{val_loss:.4f}\tval_acc:{val_acc:.4f}')
#   # test
#   results[fold] = test(testloader,criterion)[1]
#   print('--------------------------------')
#   break

# print('--------------------------------')
# sum = 0.0
# for key, value in results.items():
#   print(f'Fold {key}: {value} %')
#   sum += value
# print(f'Average: {sum/len(results.keys())} %')

# %% [markdown]
# Final Training on complete dataset

# %%
datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = True)
reset_net()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters())
for epoch in range(num_epochs):
  loss = train_epoch(datasetLoader,criterion,optimizer)
  if epoch%20==19:
    PATH = f'./final_model_{epoch}.pth'
    torch.save(net.state_dict(), PATH)
  print(f'Epoch\t{epoch+1}\tloss:{loss:.4f}')


# %%
PATH = './final_model.pth'
torch.save(net.state_dict(), PATH)

# %% [markdown]
# ### Evaluation

# %%
get_ipython().system('apt install libomp-dev')
get_ipython().system('pip install faiss')
get_ipython().system('pip install faiss-gpu')


# %%
# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

from __future__ import print_function

import os, sys
import faiss
import numpy as np

from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils import get_id
from metrics import rank1, rank5, calc_ap



# ### Set feature volume sizes (height, width, depth) 
# TODO: update with your model's feature length

batch_size = 1
H, W, D = 1, 1, 128 # for dummymodel we have feature volume 7x7x2048

# ### Load Model

# TODO: Uncomment the following lines to load the Implemented and trained Model

save_path = "final_model_199.pth"
model = SliceNet(numClasses = numClasses, inference=True)
model.load_state_dict(torch.load(save_path), strict=False)
model.eval()

# TODO: Comment out the dummy model
# model = DummyModel(batch_size, H, W, D)

# ### Data Loader for query and gallery

# TODO: For demo, we have resized to 224x224 during data augmentation
# You are free to use augmentations of your own choice
transform_query_list = [
        transforms.Resize((128,48)),
        transforms.ToTensor(),
        transforms.Normalize(meanCam, stdCam)]
transform_gallery_list = [
        transforms.Resize(size=(128,48)),
        transforms.ToTensor(),
        transforms.Normalize(meanCam, stdCam)]

data_transforms = {
        'query': transforms.Compose( transform_query_list ),
        'gallery': transforms.Compose(transform_gallery_list),
    }

image_datasets = {}
data_dir = starter_path+"/data/val"

image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'), data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'), data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

class_names = image_datasets['query'].classes


# ###  Extract Features
def extract_feature_alt(dataset):
  features =  torch.FloatTensor()
  for i in tqdm(range(len(dataset))):
    img = dataset[i][0]
    img = img[None, :]
    output = model(img)
    output = output[None, None, :]
    features = torch.cat((features, output.detach().cpu()), 0)
  return features

def extract_feature(dataloaders):
    
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    for data in tqdm(dataloaders):
        img, label = data
        # print(label)
        # Uncomment if using GPU for inference
        #img, label = img.cuda(), label.cuda()

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size
        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
    return features

# Extract Query Features

query_feature= extract_feature_alt(image_datasets['query'])

# Extract Gallery Features

gallery_feature = extract_feature_alt(image_datasets['gallery'])

# Retrieve labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


# ## Concat Averaged GELTs
concatenated_query_vectors = []
for query in tqdm(query_feature):
    fnorm = torch.norm(query, p=2, dim=0, keepdim=True)#*np.sqrt(H*W)
    query_norm = query.div(fnorm.expand_as(query))
    concatenated_query_vectors.append(query_norm.view((-1)))

concatenated_gallery_vectors = []
for gallery in tqdm(gallery_feature):
    fnorm = torch.norm(gallery, p=2, dim=0, keepdim=True)#*np.sqrt(H*W)
    gallery_norm = gallery.div(fnorm.expand_as(gallery))
    concatenated_gallery_vectors.append(gallery_norm.view((-1)))
  

# ## Calculate Similarity using FAISS

index = faiss.IndexIDMap(faiss.IndexFlatIP(H*W*D))

index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label))

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k


# ### Evaluate 

rank1_score = 0
rank5_score = 0
ap = 0
count = 0
for query, label in zip(concatenated_query_vectors, query_label):
    count += 1
    label = label
    output = search(query, k=10)
    rank1_score += rank1(label, output) 
    rank5_score += rank5(label, output) 
    print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score), end="\r")
    ap += calc_ap(label, output)

print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                             rank5_score/len(query_feature), 
                                             ap/len(query_feature)))     


