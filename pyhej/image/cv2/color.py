import cv2 as cv

image = cv.imread('tmps/img1.jpg')
image = cv.imread('tmps/img1.jpg', cv.IMREAD_COLOR) #BGR
image = cv.imread('tmps/img1.jpg', cv.IMREAD_GRAYSCALE) #GRAY

rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)


################################################################
# HSV 标记黑色区域
# 色调(H)范围[0,179],饱和度(S)范围[0,255],明度(V)范围[0,255]
# 利用HSV可以方便的查找一个指定对象的颜色,因为HSV比RGB的颜色表示要简单
# 色 | H       | S       | V
# 黑 | (0,180) | (0,255) | (  0, 46)
# 灰 | (0,180) | (0, 43) | ( 46,220)
# 白 | (0,180) | (0, 30) | (221,255)
# 比如我们可以追踪黑色物体,代码如下:
################################################################
import cv2 as cv
import numpy as np
image = cv.imread('tmps/img1.jpg')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, np.array([0,0,0]), np.array([180,255,46]))
mask = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.subplot(122), plt.imshow(mask, cmap='gray')
plt.show()