from .pillow import load_img as imread
from .pillow import img_to_array as img2arr
from .pillow import array_to_img as arr2img
from .pillow import image_new
from .pillow import draw_text
from .pillow import draw_polygon
from .pillow import draw_rectangle


import cv2 as cv
import numpy as np
from PIL import Image


def pil2cv(image):
    '''
    image = Image.open('1.jpg')
    '''
    return cv.cvtColor(np.asarray(image, dtype='uint8'), cv.COLOR_RGB2BGR)


def cv2pil(image):
    '''
    image = cv.imread('1.jpg')
    '''
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))