# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

13/11/2018 4:54 PM
"""
#encoding:utf-8

from matplotlib import pyplot as plt
import cv2
import numpy as np
image = cv2.imread("MD-Experiment-0008_p17.bmp",0)
if image is None:
    print('error opening')
cv2.imshow("Original",image)
print(image.dtype,image.shape)

hist = cv2.calcHist([image],[0],None,[256],[0,256])

image[image > 100] = 0

cv2.imshow('test',image)
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)#画图
plt.xlim([0,256])#设置x坐标轴范围
plt.show()#显示图像
