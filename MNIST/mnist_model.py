# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

10/11/2018 3:46 PM
"""

import warnings
import csv
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

train_path = './train.csv'
test_path = './test.csv'
dtype = torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('dropout',type=float,default=0.5)

args = parser.parse_args()


class MnistDataset(Dataset):

    def __init__(self, csv_path, dtype, mode):
        train_data = pd.read_csv(csv_path)
        self.dtype = dtype
        self.mode = mode
        if mode == 'train' or mode == 'val':
            labels = train_data.iloc[:, 0:1]
            pixels = train_data.iloc[:, 1:]/255
            # pixels_train, pixels_val, labels_train, labels_val = train_test_split(pixels, labels, random_state=0,
            #                                                                      train_size=0.9)
            pixels_train = pixels
            labels_train = labels
            self.N = pixels_train.shape[0]
            # self.V = pixels_val.shape[0]
            self.pixels_train = np.array(pixels_train).reshape([self.N, 1, 28, 28])
            self.labels_train = np.array(labels_train).reshape([self.N, 1])
            # self.labels_val = np.array(labels_val).reshape([self.V, 1])
            # self.pixels_val = np.array(pixels_val).reshape([self.V, 1, 28, 28])
        if mode == 'test':
            test_data = pd.read_csv(csv_path)
            test_data = test_data/255
            self.T = test_data.shape[0]
            self.test = np.array(test_data).reshape([self.T,1,28,28])

    def __getitem__(self, idx):
        if self.mode == 'train':
            label = torch.from_numpy(self.labels_train[idx]).type(self.dtype)
            img = torch.from_numpy(self.pixels_train[idx]).type(self.dtype)
            return img, label
        if self.mode == 'val':
            label = torch.from_numpy(self.labels_val[idx]).type(self.dtype)
            img = torch.from_numpy(self.pixels_val[idx]).type(self.dtype)
            return img, label
        if self.mode == 'test':
            img = torch.from_numpy(self.test[idx]).type(self.dtype)
            return img

    def __len__(self):
        if self.mode == 'train':
            return self.N
        if self.mode == 'val':
            return self.V
        if self.mode == 'test':
            return self.T


training_dataset = MnistDataset(train_path, dtype, 'train')
train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)

# val_dataset = MnistDataset(train_path, dtype, 'val')
# val_loader = DataLoader(val_dataset)

test_dataset = MnistDataset(test_path,dtype,'test')
test_loader = DataLoader(test_dataset)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


temp_model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 1
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 2
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 3
    Flatten())

temp_model = temp_model.type(dtype)
temp_model.train()
size = 0
for t, (x, y) in enumerate(train_loader):
    x_var = Variable(x.type(dtype))
    size = temp_model(x_var).size()[1]
    if (t == 0):
        break

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 1
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 2
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 3
    Flatten(),
    nn.Linear(size, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(p=args.dropout),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10),
    nn.Softmax()

)


def train(model, loss_fn, optimizer, num_epochs=2):
    for epoch in range(num_epochs):
        print("Starting epoch {0}{1}")
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(dtype)
            # y = y[:, 0].type(dtype).long()
            y = y[:,0].long()
            scores = model(x)
            loss = loss_fn(scores, y)
            if (i + 1) % 10 == 0:
                print('i=%d,loss = %.4f' % (i + 1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(model, loader):
    num_correct = 0
    num_sample = 0
    model.eval()
    for i, (x, y) in enumerate(loader):
        y = y.numpy()
        scores = model(x)
        scores = scores.data.numpy()
        preds = scores.argmax()
        num_correct += (preds == y)
        num_sample += 1
    acc = float(num_correct) / num_sample
    print("accuracy(%.2f)" % (100 * acc))


def submission(model, loader):
    model.eval()
    labels = []
    for i, x in enumerate(loader):
        if (i % 1000 == 0):
            print("prediction:" + str(i / 1000))
        scores = model(x)
        scores = scores.data.numpy()
        preds = scores.argmax()
        labels.append(preds)

    with open('solution.csv', 'w') as csvfile:
        fileds = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fileds)
        writer.writeheader()
        for i in range(len(labels)):
            if (i % 1000 == 0):
                print("writing:" + str(i / 1000))

            writer.writerow({'ImageId': str(i + 1), 'Label': labels[i]})


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
if not os.path.isfile('saved_model'):
    train(model, loss_fn, optimizer, num_epochs=)
    torch.save(model, "saved_model")
the_model = torch.load("saved_model")
# check_accuracy(the_model, val_loader)

submission(the_model, test_loader)
