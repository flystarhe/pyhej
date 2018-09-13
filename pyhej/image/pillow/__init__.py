import os
import re
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


URL_REGEX = re.compile(r'http://|https://|ftp://')
DEF_FONT = ImageFont.truetype(os.path.join(os.environ["PYHEJ_DATA"], 'datasets/fonts/DENG.TTF'), size=32)
DEF_COLOR = (0, 0, 0)


def load_img(path, mode=None, target_size=None):
    '''Loads an image into PIL format
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        path: Path to image file or url
        mode: String, must in {None, 'L', 'RGB', 'YCbCr'}
        target_size: `None` or tuple of ints `(height, width)`

    # Returns
        A PIL Image instance or `None`

    # Raises
        ..
    '''
    assert mode in {None, 'L', 'RGB', 'YCbCr'}, "mode error!"

    try:
        if URL_REGEX.match(path):
            response = requests.get(path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(path)
    except IOError:
        return None

    if mode is not None:
        if img.mode != mode:
            img = img.convert(mode)

    if target_size is not None:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)

    return img


def img_to_array(img, data_format='channels_last'):
    '''Converts a PIL Image instance to a Numpy array
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        img: PIL Image instance
        data_format: Image data format

    # Returns
        A 3D Numpy array, as `(height, width, channel)`

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    '''
    if img is None:
        return None

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    x = np.asarray(img, dtype=np.float32)

    if x.ndim == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif x.ndim == 2:
        if data_format == 'channels_first':
            x = x.reshape((1,) + x.shape)
        else:
            x = x.reshape(x.shape + (1,))
    else:
        raise ValueError('Unsupported image shape:', x.shape)

    return x


def array_to_img(x, data_format='channels_last', scale=False):
    '''Converts a 3D Numpy array to a PIL Image instance
    Notes: PIL image has format `(width, height, channel)`
           Numpy array has format `(height, width, channel)`

    # Arguments
        x: Input Numpy array
        data_format: Image data format
        scale: Whether to rescale image values to be within `[0, 255]`

    # Returns
        A PIL Image instance

    # Raises
        ValueError: if invalid `x` or `data_format` is passed
    '''
    x = x.astype(np.float32)

    if x.ndim == 2:
        x = x.reshape(x.shape + (1,))

    if x.ndim != 3:
        raise ValueError('Image array to have rank 3.', x.shape)

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)

    if scale:
        x_min = x.min()
        x_max = x.max()
        if x_max == x_min:
            x_max += 1
        x = (x-x_min) / (x_max-x_min) * 255

    if x.shape[2] == 3:
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])


def image_new(size, color=None):
    '''
    size: A 2-tuple, containing (width, height) in pixels
    color: What color to use for the image, Default is black
    '''
    if color is None:
        color = DEF_COLOR

    return Image.new('RGB', size, color)


def draw_text(img, pos, text, font=None, fill=None):
    '''
    img: PIL.Image.Image object
    pos: Top left corner of the text
    text: A text
    '''
    if font is None:
        font = DEF_FONT

    if fill is None:
        fill = DEF_COLOR

    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill=fill)

    return None


def draw_polygon(img, xys, fill=None, outline=None):
    '''
    img: PIL.Image.Image object
    xys: [(x, y), (x, y), ...] or [x, y, x, y, ...]
    '''
    draw = ImageDraw.Draw(img)
    draw.polygon(xys, fill, outline)

    return None


def draw_rectangle(img, xys, fill=None, outline=None):
    '''
    img: PIL.Image.Image object
    xys: [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
    '''
    draw = ImageDraw.Draw(img)
    draw.rectangle(xys, fill, outline)

    return None