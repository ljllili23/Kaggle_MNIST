# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

8/11/2018 3:10 PM
"""

import torch
import pandas as pd
import numpy as np
csv_file = './MNIST/train.csv'
train_rate = 0.8

mnist_frame = pd.read_csv(csv_file)
mnist_data = mnist_frame.values
np.random.shuffle(mnist_data)
# multi-dimensional arrays are only shuffled along the first axis;
data_size = mnist_data.shape[0]
train_size = int(train_rate*data_size)
train_data = mnist_data[:train_size,1:]
train_label = mnist_data[:train_size,0]
val_data = mnist_data[train_size:,1:]
val_label = mnist_data[train_size:,0]

D_int = mnist_data.shape[1]
D_out = 10
D_hidden = 

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H)
)
