import cv2 as cv
import numpy as np


"""
ImreadModes:

- cv.IMREAD_UNCHANGED: If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
- cv.IMREAD_GRAYSCALE: If set, always convert image to the single channel grayscale image (codec internal conversion).
- cv.IMREAD_COLOR: If set, always convert image to the 3 channel BGR color image.
- cv.IMREAD_ANYDEPTH: If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.

ImwriteFlags:

- cv.IMWRITE_JPEG_QUALITY: For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
- cv.IMWRITE_JPEG_PROGRESSIVE: Enable JPEG features, 0 or 1, default is False.
- cv.IMWRITE_JPEG_OPTIMIZE: Enable JPEG features, 0 or 1, default is False.
- cv.IMWRITE_JPEG_RST_INTERVAL: JPEG restart interval, 0 - 65535, default is 0 - no restart.
- cv.IMWRITE_JPEG_LUMA_QUALITY: Separate luma quality level, 0 - 100, default is 0 - don't use.
- cv.IMWRITE_JPEG_CHROMA_QUALITY: Separate chroma quality level, 0 - 100, default is 0 - don't use.
- cv.IMWRITE_PNG_COMPRESSION: For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time.
- cv.IMWRITE_PNG_STRATEGY: One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
- cv.IMWRITE_PNG_BILEVEL: Binary level PNG, 0 or 1, default is 0.
"""


def imread(file_path, flags=None):
    if flags is None:
        flags = cv.IMREAD_UNCHANGED
    return cv.imread(file_path, flags)


def imwrite(file_path, image, params=None, dtype="uint8"):
    # dtype: "uint8" or "uint16"
    # params = (cv.IMWRITE_JPEG_QUALITY, 100)
    # params = (cv.IMWRITE_JPEG_QUALITY, 100, cv.IMWRITE_PNG_COMPRESSION, 0)
    return cv.imwrite(file_path, image.astype(dtype), params)
