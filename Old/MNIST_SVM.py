# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

6/11/2018 2:13 PM
"""

from numpy import genfromtxt
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import csv
import os.path
trainingSizeRate = 0.8
# import pickle
if not os.path.isfile('svm.joblib'):
    rawTrainData = genfromtxt("./MNIST/train.csv", delimiter=',')
    print(rawTrainData.shape)
    # labels
    dataSize = rawTrainData.shape[0] - 1
    trainingSize = int(dataSize * trainingSizeRate)
    print(trainingSize)
    validationLabel = rawTrainData[trainingSize:, 0]
    validationLabel.astype(np.int8)
    print(validationLabel,validationLabel.shape)
    trainLabels = rawTrainData[1:trainingSize, 0]
    trainLabels.astype(np.int8)
    print(trainLabels,trainLabels.shape)

    # data
    rawTrainData = rawTrainData[1:, 1:]
    print(rawTrainData.shape)
    trainData = rawTrainData[:trainingSize-1,]
    trainData.astype(np.float16)
    print(trainData.shape)
    validationData = rawTrainData[trainingSize-1:, ]
    validationData.astype(np.float16)
    print(validationData.shape)

    # print(trainData[0])
    # print(trainLabels)
    # print(trainData.shape, trainData.dtype, labels.shape, labels.dtype)
    # print(trainData.shape)
    print("==training starting==")
    clf = svm.SVC(gamma='auto')
    clf.fit(trainData, trainLabels)
    joblib.dump(clf, 'svm.joblib')
else:
    rawTrainData = genfromtxt("./MNIST/train.csv", delimiter=',')
    dataSize = rawTrainData.shape[0] - 1
    trainingSize = int(dataSize * trainingSizeRate)
    print(trainingSize)
    validationLabel = rawTrainData[trainingSize:, 0]
    validationLabel.astype(np.int8)
    print(validationLabel, validationLabel.shape)
    # trainLabels = rawTrainData[1:trainingSize, 0]
    # trainLabels.astype(np.int8)
    # print(trainLabels, trainLabels.shape)

    # data
    rawTrainData = rawTrainData[1:, 1:]
    print(rawTrainData.shape)
    # trainData = rawTrainData[:trainingSize - 1, ]
    # trainData.astype(np.float16)
    # print(trainData.shape)
    validationData = rawTrainData[trainingSize - 1:, ]
    validationData.astype(np.float16)
    print(validationData.shape)
print("==validation starting==")
clf = joblib.load("svm.joblib")
result = np.zeros((0, 2), dtype=np.float16)
count = 1
for data in validationData:
    temp = [count, clf.predict(np.reshape(data, (1, -1))).item(0)]
    # print(temp)
    temp = np.array(temp,np.int64)
    count += 1
    print(temp,temp.dtype)
    result = np.row_stack((result,temp))

with open("result.csv",'wt') as fout:
    csvout = csv.writer(fout)
    csvout.writerow(['ImageId','Label'])
    csvout.writerows(result)
# print("==test starting==")
# clf = joblib.load('svm.joblib')
# testData = genfromtxt("./MNIST/test.csv", delimiter=',')
# testData = testData[1:, ]
# testData.astype(np.float16)
# print(testData.shape)
# result = np.zeros((0, 2), dtype=np.float16)
# print(result)
# for data in testData:
#     temp = clf.predict(np.reshape(data, (1, -1)))
#     print(temp)
#     # result = np.row_stack((result, temp))
#
# with open('result.csv', 'wt') as fout:
#     csvout = csv.writer(fout)
#     csvout.writerows(result)
