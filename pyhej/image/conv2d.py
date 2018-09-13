import cv2 as cv
import scipy.signal as signal


def conv2d_scp(arr, kernel):
    '''零值填补
    arr = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype='uint8')
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]], dtype='int8')
    '''
    return signal.convolve2d(arr, kernel, mode='same')


def conv2d_cv2(arr, kernel):
    '''对称填补
    arr = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype='uint8')
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]], dtype='int8')
    ddepth: -1, CV_8U: uint8, CV_16S: int16, CV_32F: float32, CV_64F: float64
      当`ddepth=-1`时,输出图像使用输入图像的数据类型,可能会产生截断
    '''
    return cv.filter2D(arr, cv.CV_16S, kernel)


'''
# Doc
https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/

# 常见核

## Mean
np.ones((3,3))/9

## Gaussian
cv.getGaussianKernel(ksize, sigma)

## Laplacian
np.array([[0, 1, 0],
          [1,-4, 1],
          [0, 1, 0]])

np.array([[1, 1, 1],
          [1,-8, 1],
          [1, 1, 1]])

## Sobel
np.array([[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]])

np.array([[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]])

## Scharr
np.array([[ -3, 0,  3],
          [-10, 0, 10],
          [ -3, 0,  3]])
'''