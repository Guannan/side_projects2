#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/ultramax-transformer.jpg',1)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image
mask_img = cv2.inRange(hsv_img, lower_blue, upper_blue)

kernel = np.ones((3,3),np.uint8)
# dilation_img = cv2.dilate(mask_img,kernel,iterations = 1)
erosion_img = cv2.erode(mask_img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)   # gets the outline of the shape

cv2.imshow('image',erosion_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('images/eroded.jpg',erosion_img)

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# plt.show()

