# http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
# skimage.measure.compare_psnr(im_true, im_test, data_range=None, dynamic_range=None)
#   im_true, im_test (ndarray): Ground-truth image. Test image.
# skimage.measure.compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, **kwargs)
#   X, Y (ndarray): Image. Any dimensionality.
from skimage.measure import compare_psnr, compare_ssim
numpy_psnr = compare_psnr
numpy_ssim = compare_ssim


################################################################
# SSIM
# pip install pyssim
################################################################
def image_ssim(A_image, B_image, size=None, cw=False):
    """
    Args:
      A_image/B_image (str or PIL.Image)
      size (tuple, optional): resize the image to `(width, height)`
      cw: compute the complex wavelet SSIM
    """
    from ssim import SSIM
    from ssim.utils import get_gaussian_kernel
    if cw:
        ssim_base = SSIM(A_image, size=size)
        ssim_value = ssim_base.cw_ssim_value(B_image)
    else:
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
        ssim_base = SSIM(A_image, gaussian_kernel_1d, size=size)
        ssim_value = ssim_base.ssim_value(B_image)
    return ssim_value


################################################################
# PSNR
################################################################
def image_psnr(A_image, B_image, peak=255., mode="L"):
    """
    Args:
      A_image/B_image (str or PIL.Image)
      peak (float): the maximum value
      mode (string): "L" or "RGB" or "YCbCr"
    """
    import math
    import numpy as np
    from PIL import Image
    if isinstance(A_image, str):
        A_image = Image.open(A_image)
    if isinstance(B_image, str):
        B_image = Image.open(B_image)
    A_image = np.asarray(A_image.convert(mode))
    B_image = np.asarray(B_image.convert(mode))
    imdff = A_image - B_image
    mse = np.mean(imdff**2)
    if mse == 0:
        return 100
    return 10 * math.log10(peak**2 / mse)