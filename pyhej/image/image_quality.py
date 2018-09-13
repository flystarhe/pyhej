'''
=> http://blog.csdn.net/real_myth/article/details/50827940
无参考图像的清晰度评价方法
'''
import cv2 as cv


def quality_of_laplacian(image_path, threshold=100):
    '''http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    http://python.jobbole.com/83702/
    这种方法凑效的原因就在于拉普拉斯算子定义本身:
      它被用来测量图片的二阶导数,突出图片中强度快速变化的区域
      和Sobel以及Scharr算子十分相似,也经常用于边缘检测
    此算法基于以下假设:
      如果图片具有较高方差,那么它就有较广的频响范围,代表着正常,聚焦准确的图片
      如果图片具有较小方差,那么它就有较窄的频响范围,意味着图片中的边缘数量很少
      正如我们所知道的,图片越模糊,其边缘就越少
    '''
    img = cv.imread(image_path, 0)
    res = cv.Laplacian(img, cv.CV_64F).var()
    return res>threshold, res


def quality_of_reblur(image_path, ksize=5):
    '''
    如果图像模糊了,对它进行一次模糊处理,高频分量变化不大
    如果原图是清晰的,对它进行一次模糊处理,高频分量变化会非常大
      待评测图像 => 高斯模糊处理 => 退化的图像 => 比较原图像和退化图像的变化情况
      - cv.blur
      - cv.GaussianBlur
      - cv.medianBlur
    '''
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_blur = cv.medianBlur(img, ksize)
    res = cv.Laplacian(img, cv.CV_64F).var()
    res_blur = cv.Laplacian(img_blur, cv.CV_64F).var()
    return (res-res_blur)/res


def quality_of_sobel(image_path, ksize=5):
    '''http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    '''
    img = cv.imread(image_path)
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
    res = (sobelx + sobely).var()
    #from matplotlib import pyplot as plt
    #plt.imshow(sobelx + sobely, cmap='gray')
    #plt.show()
    return res


'''
import cv2 as cv
from matplotlib import pyplot as plt
from pyhej.image.image_quality import quality_of_laplacian

plt.figure(figsize=(16, 16))
for i in range(4):
    image_path = '/data2/datasets/blurred-pictures/{}.jpg'.format(i+1)
    plt.subplot(2,2,i+1), plt.imshow(cv.imread(image_path, 0), cmap='gray')
    plt.title(quality_of_laplacian(image_path)), plt.xticks([]), plt.yticks([])
plt.show()
'''