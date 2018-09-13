import codecs
import h5py
import numpy as np
from PIL import Image


DEF_SIZE = 41
DEF_STRIDE = DEF_SIZE//2


def dataset_from_file(file_name, save_path, upscale=False):
    data = []
    with codecs.open(file_name, 'r', 'utf-8') as reader:
        for line in reader.readlines():
            img_b, img_h = line.strip().split(',')
            data.append((img_b, img_h))

    return dataset_from_list(data, save_path, upscale)


def dataset_from_list(data, save_path, upscale=False):
    inputs = []
    labels = []
    for img_b, img_h in data:
        try:
            img_b = Image.open(img_b)
            img_h = Image.open(img_h)

            if img_b.mode == 'L':
                img_b_y = img_b
            else:
                img_b_y, _, _ = img_b.convert('YCbCr').split()

            if img_h.mode == 'L':
                img_h_y = img_h
            else:
                img_h_y, _, _ = img_h.convert('YCbCr').split()

            if upscale:
                if img_b_y.size != img_h_y.size:
                    img_b_y = img_b_y.resize(img_h_y.size, Image.BICUBIC)

            img_b_y = np.asarray(img_b_y, dtype=np.float32)
            img_b_y = img_b_y.reshape((1,) + img_b_y.shape)
            img_b_y /= 255.

            img_h_y = np.asarray(img_h_y, dtype=np.float32)
            img_h_y = img_h_y.reshape((1,) + img_h_y.shape)
            img_h_y /= 255.
        except Exception as e:
            continue

        hs = set(tuple(range(0, img_b_y.shape[1]-DEF_SIZE, DEF_STRIDE)) + (img_b_y.shape[1]-DEF_SIZE,))
        ws = set(tuple(range(0, img_b_y.shape[2]-DEF_SIZE, DEF_STRIDE)) + (img_b_y.shape[2]-DEF_SIZE,))
        for h in hs:
            for w in ws:
                inputs.append(img_b_y[:,h:h+DEF_SIZE,w:w+DEF_SIZE])
                labels.append(img_h_y[:,h:h+DEF_SIZE,w:w+DEF_SIZE])

    hf = h5py.File(save_path, 'w')
    hf['data'] = np.array(inputs)
    hf['label'] = np.array(labels)
    hf.close()

    return save_path


def h5info(save_path):
    hf = h5py.File(save_path)
    result = {'data' : hf['data'].shape, 'label': hf['label'].shape}
    hf.close()
    return result


'''
import sys
sys.path.insert(0, '/data2/gits')

from pyhej.sr.h5 import dataset_from_file, h5info

dataset_from_file('/data2/datasets/slyx/mr2_sr_x2/dataset_2_3/dataset_train.txt',
                  '/data2/datasets/slyx/mr2_sr_x2/dataset_2_3/dataset_train.h5',
                  upscale=True)
'''