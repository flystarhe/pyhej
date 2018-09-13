'''https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
'''
import codecs
import random
import numpy as np
from keras.preprocessing.image import Iterator, ImageDataGenerator
from keras.utils import to_categorical
from pyhej.image.pillow import load_img, img_to_array


def random_transform(data_generator, x, y, seed=None):
    if seed is None:
        seed = random.randint(1, 10**6)

    x = data_generator.random_transform(x, seed)
    x = data_generator.standardize(x)

    y = data_generator.random_transform(y, seed)
    y = data_generator.standardize(y)

    return x, y


class FileIterator(Iterator):
    def __init__(self, path, data_generator, classes=None, cfg={}):
        '''
        `path` file format:
            image_uri,image_label,...
            image_uri,image_label,...
            ...
        '''
        with codecs.open(path, 'r', 'utf-8') as reader:
            lines = [line.strip().split(',') for line in reader.readlines() if len(line)>1]

        if classes is None:
            classes = list(set(line[1] for line in lines))

        self.data_generator = data_generator
        self.samples = len(lines)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        target_size = cfg.get('target_size', (256,256))
        data_format = cfg.get('data_format', 'channels_last')
        color_mode = cfg.get('color_mode', 'rgb')
        class_mode = cfg.get('class_mode', 'categorical')
        batch_size = cfg.get('batch_size', 32)
        shuffle = cfg.get('shuffle', True)
        seed = cfg.get('seed', None)

        self.target_size = tuple(target_size)
        self.data_format = data_format
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            if data_format == 'channels_last':
                self.image_shape = tuple(target_size) + (3,)
            else:
                self.image_shape = (3,) + tuple(target_size)
        else:
            if data_format == 'channels_last':
                self.image_shape = tuple(target_size) + (1,)
            else:
                self.image_shape = (1,) + tuple(target_size)
        self.class_mode = class_mode

        self.fnames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        for i, line in enumerate(lines):
            self.fnames.append(line[0])
            self.classes[i] = self.class_indices.get(line[1])
        super(FileIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype='float32')
        mode = 'L' if self.color_mode == 'grayscale' else 'RGB'
        for i, j in enumerate(index_array):
            img = load_img(self.fnames[j], mode=mode, target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.data_generator.random_transform(x)
            x = self.data_generator.standardize(x)
            batch_x[i] = x

        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = to_categorical(self.classes[index_array], self.num_classes)
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
            if isinstance(index_array, tuple):
                index_array = index_array[0]
        return self._get_batches_of_transformed_samples(index_array)