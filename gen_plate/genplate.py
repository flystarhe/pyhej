import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import cv2
import numpy as np
import math


class GenPlate(object):
    def __init__(self, font_zh=None, font_en=None, no_plates=None):
        wise_dir = os.path.dirname(__file__)
        if font_zh is None:
            font_zh = wise_dir + '/font/plate_zh.ttf'
        if font_en is None:
            font_en = wise_dir + '/font/plate_en.ttf'
        if no_plates is None:
            no_plates = wise_dir + '/no_plates'
        self.font_zh = ImageFont.truetype(font_zh, 43, 0)
        self.font_en = ImageFont.truetype(font_en, 60, 0)
        self.img = np.array(Image.new('RGB', (226,70), (255,255,255)))
        self.bg = cv2.resize(cv2.imread(wise_dir + '/template.bmp'), (226,70))
        self.no_plates_path = []
        for dirpath, dirnames, filenames in os.walk(no_plates):
            for filename in filenames:
                self.no_plates_path.append(dirpath + '/' + filename)

    def draw(self, val):
        offset = 2
        self.img[0:70, offset+8:offset+8+23] = gen_zh(self.font_zh, val[0])
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = gen_en(self.font_en, val[1])
        for i in range(5):
            base = offset+8+23+6+23+17+i*23+i*6
            self.img[0:70, base:base+23] = gen_en(self.font_en, val[2+i])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            fg = self.draw(text)
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg, self.bg)
            com = rot(com, r(60)-30, com.shape, 30)
            com = rot_randrom(com, 10, (com.shape[1], com.shape[0]))
            com = tfactor(com)
            com = random_envirment(com, self.no_plates_path)
            com = add_gauss(com, 1+r(4))
            com = add_noise(com)
            return com

    def gen_text(self, pos, val):
        text = ''
        for i in range(7):
            if i == pos:
                text += val
            else:
                if i == 0:
                    text += chars[r(31)]
                elif i == 1:
                    text += chars[41+r(24)]
                else:
                    text += chars[31+r(34)]
        return text

    def gen_batch(self, batch_size=1, output_path=None, size=(272, 72)):
        '''
        OpenCV默认颜色通道为BGR
        RGB请自行转换: img = img[:,:,::-1]
        '''
        txts, imgs = [], []
        for i in range(batch_size):
            txt = self.gen_text(-1, -1)
            img = self.generate(txt)
            txts.append(txt)
            imgs.append(cv2.resize(img, size))
        if output_path is not None:
            if not os.path.isdir(output_path):
                os.mkdirs(output_path)
            for i, img in enumerate(imgs):
                cv2.imwrite('%s/%03d.jpg' % (output_path, i), img)
        return (txts, imgs)


index = {'京': 0, '沪': 1, '津': 2, '渝': 3, '冀': 4, '晋': 5, '蒙': 6, '辽': 7, '吉': 8, '黑': 9, '苏': 10, '浙': 11, '皖': 12,
         '闽': 13, '赣': 14, '鲁': 15, '豫': 16, '鄂': 17, '湘': 18, '粤': 19, '桂': 20, '琼': 21, '川': 22, '贵': 23, '云': 24,
         '藏': 25, '陕': 26, '甘': 27, '青': 28, '宁': 29, '新': 30, '0': 31, '1': 32, '2': 33, '3': 34, '4': 35, '5': 36,
         '6': 37, '7': 38, '8': 39, '9': 40, 'A': 41, 'B': 42, 'C': 43, 'D': 44, 'E': 45, 'F': 46, 'G': 47, 'H': 48,
         'J': 49, 'K': 50, 'L': 51, 'M': 52, 'N': 53, 'P': 54, 'Q': 55, 'R': 56, 'S': 57, 'T': 58, 'U': 59, 'V': 60,
         'W': 61, 'X': 62, 'Y': 63, 'Z': 64}

chars = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂',
         '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
         'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def rot(img, angel, shape, max_angel):
    '''
    使图像轻微的畸变
        img: 输入图像
        factor: 畸变的参数
        size: 为图片的目标尺寸
    '''
    size_o = [shape[1], shape[0]]
    size = (shape[1]+int(shape[0]*math.cos((float(max_angel)/180)*3.14)), shape[0])
    interval = abs(int(math.sin((float(angel)/180)*3.14)*shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if angel > 0:
        pts2 = np.float32([[interval,0],[0,size[1]],[size[0],0],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]],[size[0]-interval,0],[size[0],size_o[1]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def rot_randrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0,0],[0,shape[0]],[shape[1],0],[shape[1],shape[0]]])
    pts2 = np.float32([[r(factor),r(factor)],[r(factor),shape[0]-r(factor)],[shape[1]-r(factor),r(factor)],
                       [shape[1]-r(factor),shape[0]-r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = hsv[:,:,0]*(0.8 + np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3 + np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2 + np.random.random()*0.8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_envirment(img, data_set):
    index = r(len(data_set))

    env = cv2.imread(data_set[index])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))

    bak = (img==0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img


def gen_zh(f, val):
    img = Image.new('RGB', (45,70), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0,3), val, (0,0,0), font=f)
    img = img.resize((23,70))
    return np.array(img)


def gen_en(f, val):
    img = Image.new('RGB', (23,70), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0,2), val, (0,0,0), font=f)
    return np.array(img)


def add_gauss(img, level):
    return cv2.blur(img, (level*2 + 1, level*2 + 1))


def r(val):
    return int(np.random.random()*val)


def add_noise_single_channel(single):
    diff = 255-single.max()
    noise = np.random.normal(0, 1+r(6), single.shape)
    noise = (noise-noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst


def add_noise(img,sdev = 0.5,avg=10):
    img[:,:,0] = add_noise_single_channel(img[:,:,0])
    img[:,:,1] = add_noise_single_channel(img[:,:,1])
    img[:,:,2] = add_noise_single_channel(img[:,:,2])
    return img