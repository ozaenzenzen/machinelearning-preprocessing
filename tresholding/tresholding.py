# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:10:11 2021

@author: baraya
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

src = 'IMG_20211111_133520.jpg';

#src = 'C:/Users/baraya/Documents/Ozan/Kuliah/Semester 7/Data Skripsi/Data Lama ver 2/Dataset/Kibon Junbi Seogi/IMG_20211125_061714.jpg';

img = cv2.imread(src)

gbr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#cv2.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

retval, thres1 = cv2.threshold(img, 110, 230, cv2.THRESH_BINARY)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img_gray,5)

thres2 = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 5)

plt.xticks([]), plt.yticks([])
# plt.imshow(img_gray, cmap='gray')
# plt.imshow(img_blur, cmap='gray')
plt.imshow(thres2, cmap='gray')
plt.title('Test Thresholding')
plt.show()
