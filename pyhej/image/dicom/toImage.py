import numpy as np
from PIL import Image
import SimpleITK as sitk


def dcm2img(dcm):
    itk_img = sitk.ReadImage(dcm)
    array = sitk.GetArrayFromImage(itk_img)[0]
    array = (array - array.min()) / (array.max() - array.min())
    array = np.clip(array * 255, 0, 255)
    return Image.fromarray(array.astype("uint8"))


def test():
    import os
    import glob
    output_dir = "tmps/03141622"
    os.makedirs(output_dir, exist_ok=True)
    dcm_list = glob.glob("/data2/datasets/tmps/19030516/05030000/*")
    for num, dcm in enumerate(dcm_list, 1):
        try:
            name = os.path.basename(dcm).split(".")[0]
            img_path = os.path.join(output_dir, "{:04d}_{}.png".format(num, name))
            dcm2img(dcm).save(img_path)
        except Exception as e:
            print("!Error:", num, dcm)