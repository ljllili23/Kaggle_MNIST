# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

9/11/2018 10:14 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import datetime


class mnistDataset(Dataset):
    """mnist dataset."""
    def __init__(self,csv_file,transform=None):
        self.mnist_frame = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.mnist_frame)

    def __getitem__(self,idx):
        mnist_data = self.mnist_frame.iloc[idx,1:].values
        mnist_data = mnist_data.astype('float32').reshape(28,28)
        # mnist_data = mnist_data.astype('float')
        mnist_label = self.mnist_frame.iloc[idx,0]
        sample = {'image':mnist_data, 'label':mnist_label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""
    def __call__(self,sample):
        image, label = sample['data'],sample['label']
        return {'image':torch.from_numpy(image),
                'label':label}



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())


dataset = mnistDataset(csv_file='train.csv',transform=transforms.Compose([ToTensor()]))
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_dataset,test_dataset = random_split(dataset,[train_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=20,shuffle=True)
print(train_dataloader)
for i_batch, sample_batched in enumerate(train_dataloader):
    print(sample_batched['image'].dtype, sample_batched['image'].size(),sample_batched['image'].shape)
    out = net(input)
    print(out)