'''https://github.com/fchollet/keras/blob/master/keras/applications
'''
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from pyhej.image import load_img, img_to_array


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    '''Checks if a file is an image
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    '''
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_x_mean(root, mode='pixel', target_size=None):
    '''
    # Arguments
        root: image root dir
        mode: 'pixel' or 'channel'
        target_size: if `mode` is 'pixel', must set `target_size`
    # Returns
        Preprocessed tensor
    '''
    assert mode in {'pixel', 'channel'}
    fpaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if is_image_file(filename):
                fpaths.append(os.path.join(dirpath, filename))
    xs_sum = None
    for filepath in fpaths:
        img = load_img(filepath, target_size=target_size)
        arr = img_to_array(img, data_format='channels_last')
        if 'pixel' == mode:
            xs_val = arr/255.
        else:
            xs_val = arr/255.
            xs_val = np.mean(arr, axis=(0, 1))
            xs_val = np.reshape(xs_val, (1, 1, 3))
        if xs_sum is None:
            xs_sum = xs_val
        else:
            xs_sum += xs_val
    return xs_sum/len(fpaths)


def preprocess_input(x, x_mean=None):
    '''Preprocesses a tensor encoding a batch of images
    # Arguments
        x_mean: from `get_x_mean`
        x: input Numpy tensor, 4D
    # Returns
        Preprocessed tensor
    '''
    if x_mean is None:
        x /= 255.
        x -= 0.5
        x *= 2.
    else:
        x /= 255.
        x -= x_mean
    return x


datagen_args = dict(
    rotation_range=40,  # randomly rotate images in the range (deg 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally
    height_shift_range=0.2,  # randomly shift images vertically
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    preprocessing_function=preprocess_input)


def loader_from_directory(root, datagen=None, target_size=(32, 32), batch_size=32, shuffle=True, seed=None):
    if datagen is None:
        datagen = ImageDataGenerator(**datagen_args)
    elif isinstance(datagen, dict):
        datagen = ImageDataGenerator(**datagen)
    return DirectoryIterator(root, datagen, target_size=target_size, batch_size=batch_size, shuffle=shuffle, seed=seed)