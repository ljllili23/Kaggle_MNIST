# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

8/11/2018 10:16 AM
"""
import os.path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
filename = './MNIST/train.csv'
training_size_rate = 0.8


mnist_frame = pd.read_csv(filename)
mnist_data = mnist_frame.values[:,1:]
print(mnist_data.shape,mnist_data.dtype)
mnist_label = mnist_frame.values[:,0]
print(mnist_label.shape,mnist_label.dtype)
data_size = mnist_data.shape[0]
training_size = int(data_size*training_size_rate)
print(training_size)
mnist_train = mnist_data[:training_size,:]
print(mnist_train.shape)
mnist_train_label = mnist_label[:training_size]
mnist_validation = mnist_data[training_size:,:]
mnist_validation_label = mnist_label[training_size:]


if not os.path.isfile('svm.joblib'):
    clf = svm.SVC()
    clf.fit(mnist_train,mnist_train_label)
    joblib.dump(clf,'svm.joblib')
else:
    clf = joblib.load('svm.joblib')

for data in mnist_validation:
    print(data)
    predict_label = clf.predict(np.reshape(data,(1,-1)))
    print(predict_label)