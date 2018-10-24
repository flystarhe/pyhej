#pip install pydicom
import os
import pydicom
import numpy as np
import matlab.engine


matlab_eng = None


def dicom_read_ct(file_path):
    plan = pydicom.dcmread(file_path, force=True)
    dtype = "" if getattr(plan, "PixelRepresentation", 0) else "u"
    dtype = "{}int{}".format(dtype, getattr(plan, "BitsAllocated", 8))
    data = np.frombuffer(plan.PixelData, dtype=dtype)
    data = data.reshape((-1, plan.Rows, plan.Columns))[:1]
    rescale_slope = getattr(plan, "RescaleSlope", 1)
    rescale_intercept = getattr(plan, "RescaleIntercept", 0)
    r_ct = data * rescale_slope + rescale_intercept
    return r_ct.clip(-1000, None)


def dicom_read_ac(file_path):
    #transfer the hunsfield units to attenuation coefficents
    r_ct = dicom_read_ct(file_path)
    #0.17 is the attenuation coef (1/cm) of water at 100 keV
    r_ac = 0.17 * r_ct / 1000 + 0.17
    return r_ac.clip(0, None)


def ac2ct(ac, window=None):
    # window is tuple, such as `(center, width)`
    ct = (ac - 0.17) * 1000 / 0.17
    if window is not None:
        center, width = window
        ct = (ct - center + width/2) / width * 255
        ct = ct.clip(0, 255).astype("uint8")
    return ct


def add_poisson_noise(ac, N=30000):
    """
    Args:
      ac: attenuation coefficients of a single ct slice
      N : x-ray source intensity, the lower number the higher noise
    Ex:
      from pyhej.image import dicom
      file_path = "/data2/tmps/SAGAN/poisson_noise_simulation/000048.dcm"
      ac = dicom.dicom_read_ac(file_path)
      print(ac.min(), ac.max())
      for N in [300000, 30000, 3000, 300]:
          ac_noise = dicom.add_poisson_noise(ac, N)
          print(ac_noise.min(), ac_noise.max())
          import numpy as np
          from PIL import Image
          from IPython.display import display
          image_A = dicom.ac2ct(ac[0], (300,2000))
          image_B = dicom.ac2ct(ac_noise[0], (300,2000))
          image_pil = Image.fromarray(np.hstack((image_A, image_B)))
          display(image_pil)
    """
    global matlab_eng
    if matlab_eng is None:
        newpath = os.path.dirname(os.path.realpath(__file__))
        matlab_eng = matlab.engine.start_matlab()
        matlab_eng.userpath(newpath, nargout=0)
        print(" * matlab userpath:", newpath)
    ac_mat = matlab.double(ac[0].tolist())
    ac_mat_noise = matlab_eng.add_poisson_noise(ac_mat, float(N))
    return np.asarray(ac_mat_noise).reshape(ac.shape).clip(0, None)


def dcmread(file_path):
    """read dicom file info
    file_path: a string, such as "your/path/file_path.dcm"
    """
    try:
        plan = pydicom.dcmread(file_path, force=True)
    except Exception as e:
        plan = None
    return plan


def pixel_array(file_path, simple=True):
    plan = pydicom.dcmread(file_path, force=True)
    dtype = "" if getattr(plan, "PixelRepresentation", 0) else "u"
    dtype = "{}int{}".format(dtype, getattr(plan, "BitsAllocated", 8))
    data = np.frombuffer(plan.PixelData, dtype=dtype)
    data = data.reshape((-1, plan.Rows, plan.Columns))[:1]
    rescale_slope = getattr(plan, "RescaleSlope", 1)
    rescale_intercept = getattr(plan, "RescaleIntercept", 0)
    r_ct = data * rescale_slope + rescale_intercept
    if not simple:
        return r_ct.clip(-1000, None), plan
    return r_ct.clip(-1000, None)


def toimage(file_path, window=None, RescaleMode="Line"):
    """
    file_path: a string, the dicom file path
    window: string or tuple, `"guess"|(low,hig)`
    RescaleMode: a string, "Line|.."
    """
    data, plan = pixel_array(file_path, simple=False)
    data = data.astype(np.float32)

    low, hig = data.min(), data.max()
    if isinstance(window, tuple):
        low, hig = window
    elif window == "guess":
        low, hig = dicom_window(plan, low, hig)
    elif hasattr(plan, "PixelPaddingValue"):
        low = getattr(plan, "PixelPaddingValue")

    data = np.clip(data, low, hig)

    if RescaleMode == "Line":
        data = (data - low)/(hig - low)
        data = np.clip(data*255, 0, 255).astype(np.uint8)
    return data, low, hig


def dicom_window(plan, low=0, hig=None):
    if hasattr(plan, "WindowCenter") and hasattr(plan, "WindowWidth"):
        window_center = getattr(plan, "WindowCenter")
        window_width = getattr(plan, "WindowWidth")
        if isinstance(window_center, pydicom.multival.MultiValue):
            low, hig = [], []
            for x, y in zip(window_center, window_width):
                r = y//2
                low.append(x - r)
                hig.append(x - r + y)
            low = min(low)
            hig = max(hig)
        else:
            r = window_width//2
            low = window_center - r
            hig = window_center - r + y
    return low, hig


def update_pixelData(file_path, data, dtype="pixel", savepath=None):
    """
    file_path: a string, the file path
    data: a numpy array
    """
    ds = pydicom.dcmread(file_path, force=True)
    if dtype != "pixel":
        rescale_slope = getattr(ds, "RescaleSlope", 1)
        rescale_intercept = getattr(ds, "RescaleIntercept", 0)
        data = (data - rescale_intercept) / rescale_slope
        data = data.astype("int16")
    # copy the data back to the original data set
    ds.PixelData = data.tostring()
    # update the information regarding the shape of the data array
    ds.Rows, ds.Columns = data.shape

    tags = data.dtype.name.split("int")
    ds.BitsAllocated = int(tags[1])
    if tags[0] == "u":
        ds.PixelRepresentation = 0
    else:
        ds.PixelRepresentation = 1

    if savepath:
        ds.save_as(savepath)
        return savepath
    return ds


def display_dir(root, pattern="**/*.dcm", cols=["Modality", "StudyDescription"]):
    """
    cols: "Modality", "StudyDescription", "NumberOfFrames"
    """
    import os
    import glob
    import pydicom
    data = []
    filepaths = glob.glob(os.path.join(root, pattern), recursive=True)
    for filepath in sorted(filepaths):
        plan = pydicom.dcmread(filepath, force=True)
        data.append([filepath] + [getattr(plan, col, None) for col in cols])
    print("=> {}, counts={}".format(root, len(data)))
    print("import pandas as pd")
    print("temp = pd.DataFrame(data, columns=['path'] + cols)")
    print("pd.pivot_table(temp, index='col1', columns='col2', aggfunc='size')")
    print("temp['col1'].value_counts()")
    return data


def display_dir_jupyter(root, pattern="**/*.dcm", window=None, attr_name="Modality", attr_value=None, limit=5):
    """
    Args:
      attr_value (string): "OT", "CT", "XA"
      limit (int, string): sample 5 or ":5" or "-5:"
    """
    import os
    import glob
    import random
    import pydicom
    from IPython.display import display
    from pyhej.image import arr2img
    filepaths = glob.glob(os.path.join(root, pattern), recursive=True)
    if isinstance(limit, int):
        filepaths = random.sample(filepaths, min(limit, len(filepaths)))
    elif isinstance(limit, str):
        if limit.startswith(":"):
            filepaths = filepaths[:int(limit[1:])]
        else:
            filepaths = filepaths[int(limit[:-1]):]
    else:
        raise ValueError("Unknown limit:", limit)
    for filepath in filepaths:
        try:
            plan = pydicom.dcmread(filepath, force=True)
            temp = getattr(plan, attr_name, "None")
            if attr_value is None or temp == attr_value:
                arr, low, hig = toimage(filepath, window)
                attr_mod = getattr(plan, "Modality", "None")
                attr_num = getattr(plan, "NumberOfFrames", 1)
                print("=> {}, modality={}, frames={}, [{},{}]".format(filepath, attr_mod, attr_num, low, hig))
                if arr.ndim > 2:
                    for tmp in arr:
                        display(arr2img(tmp))
                else:
                    display(arr2img(arr))
        except Exception as e:
            print("!!!{}, {}".format(filepath, e))