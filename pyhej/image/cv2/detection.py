## edge
'''
- https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
image = cv.imread('tmps/img1.jpg', 0)
edges = cv.Canny(image, 100, 200)
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


## line
'''
- https://docs.opencv.org/master/d6/d10/tutorial_py_houghlines.html

第二和第三个参数距离和角度的精度
第四个参数是门槛,这意味着最低的投票应该被视为一条线
- 得票数取决于线上的点数,所以它代表了应该检测到的行的最小长度
'''
import cv2 as cv
import numpy as np
image = cv.imread('tmps/img1.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 200, apertureSize=3)
lines = cv.HoughLines(edges, 1, np.pi/180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(edges, cmap='gray')
plt.subplot(122), plt.imshow(image, cmap='gray')
plt.show()


'''
HoughLinesP:
- minLineLength: 行的最小长度,比这更短的线段被拒绝
- maxLineGap: 线段之间允许的最大间距,将它们视为单行

最好的是,它直接返回行的两个端点
'''
import cv2 as cv
import numpy as np
image = cv.imread('tmps/img1.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 200, apertureSize=3)
lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1,y1), (x2,y2), (0,255,0), 2)
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(edges, cmap='gray')
plt.subplot(122), plt.imshow(image, cmap='gray')
plt.show()


## circle
'''
- https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
'''
import numpy as np
import cv2 as cv
img = cv.imread('tmps/img1.jpg', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0],i[1]), 2, (0,0,255), 3)
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(cimg, cmap='gray')
plt.show()


## sceneText
'''
- https://docs.opencv.org/3.3.1/da/d56/group__text__detect.html
- https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/textdetection.py

类特定极值区域背后的主要思想类似于MSER
其从图像的整个分量树中选择合适的极值区域(ER)
然而,这种技术与MSER的不同之处在于,通过训练用于字符检测的顺序分类器来完成对合适的ER的选择
降低MSER的稳定性要求并选择类别特定的(不一定是稳定的)区域
'''
import cv2 as cv
import numpy as np

img = cv.imread('tmps/img1.jpg')
vis = img.copy()

# Extract channels to be processed individually
channels = cv.text.computeNMChannels(img)
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels)-1
for c in range(0, cn):
    channels.append((255-channels[c]))

# Apply the default cascade classifier to each independent channel (could be done in parallel)
print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
print("    (...) this may take a while (...)")
for channel in channels:
    erc1 = cv.text.loadClassifierNM1('tmps/opencv_contrib/modules/text/samples/trained_classifierNM1.xml')
    er1 = cv.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

    erc2 = cv.text.loadClassifierNM2('tmps/opencv_contrib/modules/text/samples/trained_classifierNM2.xml')
    er2 = cv.text.createERFilterNM2(erc2, 0.5)

    regions = cv.text.detectRegions(channel, er1, er2)

    rects = cv.text.erGrouping(img, channel, [r.tolist() for r in regions])

    #Visualization
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
        cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)


## sceneText(deep)
img = cv.imread('tmps/img1.jpg')
textSpotter = cv.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
rects, outProbs = textSpotter.detect(img)
vis = img.copy()
thres = 0.6

for r in range(np.shape(rects)[0]):
    if outProbs[r] > thres:
        rect = rects[r]
        cv.rectangle(vis, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)


## 角点检测
'''
- https://docs.opencv.org/3.3.1/df/dd2/tutorial_py_surf_intro.html
- https://docs.opencv.org/3.3.1/df/d0c/tutorial_py_fast.html
- https://docs.opencv.org/3.3.1/d1/d89/tutorial_py_orb.html
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('tmps/img1.jpg', 0)
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img, None)
print(len(kp)) #699
print(surf.getHessianThreshold()) #400.0
surf.setHessianThreshold(50000) #better to have a value 300-500
kp, des = surf.detectAndCompute(img, None)
print(len(kp)) #47
img2 = cv.drawKeypoints(img, kp, None, (255,0,0), 4)
plt.figure(figsize=(18, 18)), plt.imshow(img2), plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('tmps/img1.jpg', 0)
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
plt.figure(figsize=(18, 18))
plt.subplot(121), plt.imshow(img2, cmap='gray')
plt.subplot(122), plt.imshow(img3, cmap='gray')
plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('tmps/img1.jpg', 0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.figure(figsize=(18, 18)), plt.imshow(img2), plt.show()