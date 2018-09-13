"""https://github.com/HelgeEgil/NoisePowerSpectrumFromCT"""
import pydicom
import numpy as np
import scipy.optimize as optimize


def func(data, c1, c2, c3, c4, c5, c6):
    """Helper function for 2D polynomial fit."""
    return c1 + c2*data[:,0] + c3*data[:,1] + c4 * data[:,0] ** 2 + c5 * data[:,1] ** 2 + c6 * data[:,0] * data[:,1]


def easyfunc(x, y, c1, c2, c3, c4, c5, c6):
    """2D fit function."""
    return c1 + c2*x + c3*y + c4*x**2 + c5*y**2 + c6*x*y


def poly_sub(img):
    """Subtracts 2D polynomial from img.
    Original form: z = imgROI[x,y]. create roi_size^2 number of tuples with 
    values (x, y, z). Then to feed them to scipy.curve_fit."""
    img_size = np.shape(img)[0]
    xx, yy = np.meshgrid(range(img_size), range(img_size))
    xx, yy = xx.flatten(), yy.flatten()
    zz = img.flatten()
    # like zip:
    # make (a1,b2,c3), (a2,b2,c2,),... from ((a1, a2, ...), (b1, b2, ...), ...)
    img_array = np.dstack((xx, yy, zz))[0]
    guess = [0, 0, 0, 0, 0, 0]
    params, pcov = optimize.curve_fit(func, img_array[:,:2], img_array[:,2], guess)
    # we need array of params: [1,2,3] = [[1,2,3], [1,2,3], [1,2,3], ...]
    p = np.resize(params, (img_size**2, len(params)))
    subtract_map = map(easyfunc, xx, yy, p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5])
    subtract_img = np.reshape(list(subtract_map), (img_size, img_size))
    #plt.imshow(img + np.multiply(5, subtract_img), interpolation = "nearest", cmap=cm.gray)
    #plt.savefig("original_enhanced.png")
    return img - subtract_img


def noisePowerRadAv(mat):
    """对所有相同的角尺度求平均值,得到一维的结果"""
    x, y = mat.shape
    aran = np.arange(x)
    theta = np.linspace(0, np.pi/2.0, x*2, endpoint=False)
    r = np.zeros(x)
    for ang in theta:
        yran = np.rint(aran*np.sin(ang))
        xran = np.rint(aran*np.cos(ang))
        r += np.ravel(mat[[xran.astype("int64")], [yran.astype("int64")]])
    return r/float(x)


def nps_roi(mat, sample_spacing, options={}):
    data_roi = mat.astype(float)
    roi_size = data_roi.shape[0]

    if options.get("dcCorrection", False):
        data_roi = data_roi - np.mean(data_roi)

    if options.get("2dpoly", False):
        data_roi = poly_sub(data_roi)

    if options.get("zeropad", False):
        zeropad = options.get("zeropad", 3)
        ROIzeropad = np.zeros((roi_size*zeropad, roi_size*zeropad))
        ROIzeropad[roi_size:2*roi_size, roi_size:2*roi_size] = data_roi*zeropad
        data_roi = ROIzeropad
        roi_size = data_roi.shape[0]

    FFT = np.square(np.abs(np.fft.fft2(data_roi))) * (sample_spacing**2/roi_size**2)
    fft = np.fft.fftfreq(roi_size, d=sample_spacing)

    return fft, FFT


def nps(filename, options={}):
    """
    filename = "/data2/tmps/tmps/images/IM2"
    options = {"nROI":64, "dcCorrection":True, "2dpoly":True, "zeropad":3, "nnps":True}
    xy = nps(filename, options)
    import matplotlib.pyplot as plt
    plt.plot(xy["x"], xy["y"])
    """
    plan = pydicom.dcmread(filename, force=True)
    sample_spacing = float(plan.PixelSpacing[0])
    data = plan.pixel_array

    a = float(getattr(plan, "RescaleSlope", 1))
    b = float(getattr(plan, "RescaleIntercept", 0))
    data = data * a + b

    if options.get("ROI", False):
        roi_beg, roi_end = options["ROI"]
        data_roi = data[roi_beg:roi_end, roi_beg:roi_end]
    else:
        data_roi = data

    if options.get("nROI", False):
        # nROI: 9, 16, 64, 256
        nstep = int(np.sqrt(options["nROI"]))
        roi_size = data_roi.shape[0]//(nstep-1)
        roi_items = np.linspace(0, data_roi.shape[0]//2, nstep, dtype=int)
        ffts, FFTs = [], []
        for i in roi_items:
            for j in roi_items:
                fft, FFT = nps_roi(data_roi[i:i+roi_size, j:j+roi_size], sample_spacing, options)
                ffts.append(fft)
                FFTs.append(FFT)
        fft = sum(ffts) / len(ffts)
        FFT = sum(FFTs) / len(FFTs)
    else:
        fft, FFT = nps_roi(data_roi, sample_spacing, options)

    FFT = noisePowerRadAv(FFT)

    idx = np.where(fft>0)
    noiseplot = dict(x=fft[idx], y=FFT[idx])

    if options.get("nnps", False):
        x_with = noiseplot["x"][1] - noiseplot["x"][0]
        noiseplot["y"] = noiseplot["y"] / (np.sum(noiseplot["y"]) * x_with)

    return noiseplot