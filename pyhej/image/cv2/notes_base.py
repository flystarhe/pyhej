import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

img = cv.imread(img, 0)
row, col = np.indices(img.shape)

plt.figure(figsize=(18, 18))
ax = plt.subplot(111)
ax.contourf(col, row, img, 10, alpha=0.6, cmap=plt.cm.hot)
C = ax.contour(col, row, img, 10)
ax.clabel(C, inline=True, fontsize=10)
plt.show()


################################################################
# 图像分析
# https://docs.opencv.org/3.3.1/d1/db7/tutorial_py_histogram_begins.html
################################################################
import cv2 as cv
import imutils

image = cv.imread(filepath+'/img1.jpg')

from matplotlib import pyplot as plt
plt.imshow(imutils.opencv2matplotlib(newImage))
plt.show()

image = cv.imread(filepath+'/img1.jpg', 0)

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(image, 'gray')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.hist(image[image>0].ravel(), 256)
plt.xticks([]), plt.yticks([])
plt.show()


################################################################
# 图像二值化
# https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
################################################################
import cv2 as cv

image = cv.imread(filepath+'/img1.jpg', 0)

# 一般二值化
th1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]
# 自适应二值化
image = cv.medianBlur(image, 5)
th1 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th1 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# 双峰图像二值化
image = cv.GaussianBlur(image, (5, 5), 0)
th1 = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

import matplotlib.pyplot as plt
plt.figure(figsize=(18, 18))
plt.imshow(th1, 'gray')
plt.axis('off')
plt.show()


################################################################
# 图像梯度
# https://docs.opencv.org/3.3.1/d5/d0f/tutorial_py_gradients.html
################################################################
import cv2 as cv

image = cv.imread(filepath+'/img1.jpg', 0)

gd1 = cv.Laplacian(image, cv.CV_64F)
gd1 = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
gd1 = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
gd1 = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=5)

import matplotlib.pyplot as plt
plt.figure(figsize=(18, 18))
plt.imshow(gd1, 'gray')
plt.axis('off')
plt.show()


################################################################
# 边缘检测
# https://docs.opencv.org/3.3.1/da/d22/tutorial_py_canny.html
################################################################
import cv2 as cv

image = cv.imread(filepath+'/img1.jpg', 0)
edges = cv.Canny(image, 100, 200)

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(image, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


################################################################
# 形态转化:侵蚀,扩张,开放,关闭
# https://docs.opencv.org/3.3.1/d9/d61/tutorial_py_morphological_ops.html
################################################################
import cv2 as cv
import numpy as np

image = cv.imread(filepath+'/img1.jpg', 0)

kernel = np.ones((3, 3), np.uint8)

# 腐蚀
erosion = cv.erode(image, kernel, iterations=1)
# 膨胀
dilation = cv.dilate(image, kernel, iterations=1)
# 开运算:侵蚀之后扩张,它有助于消除噪音
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
# 闭运算:扩张之后侵蚀,在关闭前景物体内的小孔或物体上的小黑点时非常有用
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.subplot(221), plt.imshow(erosion, 'gray')
plt.title('erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(dilation, 'gray')
plt.title('dilation'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(opening, 'gray')
plt.title('opening'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(closing, 'gray')
plt.title('closing'), plt.xticks([]), plt.yticks([])
plt.show()


################################################################
# 图像模糊/图像平滑
# https://docs.opencv.org/3.3.1/d4/d13/tutorial_py_filtering.html
################################################################
import cv2 as cv

image = cv.imread(filepath+'/img1.jpg', 0)

# Averaging: function cv.blur() or cv.boxFilter()
blur = cv.blur(image, (5, 5))
# Gaussian Blurring
blur = cv.GaussianBlur(image, (5, 5), 0)
# Median Blurring
blur = cv.medianBlur(image, 5)
# Bilateral Filtering
blur = cv.bilateralFilter(image, 9, 75, 75)

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(image, 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur, 'gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])
plt.show()


################################################################
# 直方图均衡/改善图像的对比度
# https://docs.opencv.org/3.3.1/d5/daf/tutorial_py_histogram_equalization.html
################################################################
import cv2 as cv
import numpy as np

image = cv.imread(filepath+'/img1.jpg', 0)

hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.plot(cdf_normalized, color='b')
plt.hist(image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

# 直方图均衡
equ = cv.equalizeHist(image)
res = np.hstack((image, equ)) # stacking images side-by-side
# 自适应直方图均衡
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equ = clahe.apply(image)
res = np.hstack((image, equ)) # stacking images side-by-side

from matplotlib import pyplot as plt
plt.figure(figsize=(18, 18))
plt.imshow(res, 'gray')
plt.axis('off')
plt.show()