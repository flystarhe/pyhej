import h5py
import numpy as np


def h5read(fpath):
    rs = []
    h5 = h5py.File(fpath, 'r')
    for key in h5.keys():
        rs.append((key, np.array(h5.get(key))))
    h5.close()
    return rs


def h5write(fpath, keys, vals):
    h5 = h5py.File(fpath, 'w')
    for key, val in zip(keys, vals):
        h5[key] = np.array(val)
    h5.close()
    return fpath


def h5info(fpath):
    rs = []
    h5 = h5py.File(fpath, 'r')
    for key in h5.keys():
        rs.append((key, np.array(h5.get(key)).shape))
    h5.close()
    return rs