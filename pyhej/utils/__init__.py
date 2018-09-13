import re
import os
import shutil
from pathlib import Path


FILE_REGEX = re.compile("[^.0-9a-zA-Z]+", re.U)


def get_fname(obj):
    """
    # Arguments
        obj: String, file path or uri

    # Returns
        A string, standard file name
    """
    return FILE_REGEX.sub("_", str(obj)).lower()


def get_path_relative(a, b, simple=True):
    """
    # Arguments
        a: String or Path object to file path
        b: String or Path object to dir path

    # Returns
        String or Path
    """
    c = Path(a).relative_to(b)
    if simple:
        return c.as_posix()
    return c


def set_dir(target_dir, mode="777", rm=False):
    xpath = Path()
    for part in Path(target_dir).parts:
        xpath /= part
        if not xpath.exists():
            os.system("mkdir -m {} {}".format(mode, str(xpath)))

    if rm:
        shutil.rmtree(str(xpath))
        os.system("mkdir -m {} {}".format(mode, str(xpath)))

    return str(xpath)


def set_parent(target_path, mode="777", rm=False):
    target_dir = os.path.dirname(target_path)
    return set_dir(target_dir, mode, rm)


def folder_split(root, size=100, pattern="*.jpg"):
    logs = []
    root = Path(root)
    data = sorted(root.glob(pattern))
    targ = "{}_/{}".format(root, root.name)
    set_parent(targ, "777", rm=True)
    for i, item in enumerate(data):
        a, b = divmod(i, size)
        temp = "{}_{:03d}/{}".format(targ, a+1, item.name)
        if b == 0:
            set_parent(temp, "777", rm=False)
        shutil.copyfile(item.as_posix(), temp)
        logs.append((i, a, b, temp))
    return logs


def split_list(n, size=None, data=None):
    pos, res = 0, []
    if size is None:
        size = len(data)
    a, b = divmod(size, n)
    for i in range(n):
        tmp = a + 1 if i < b else a
        res.append((pos, pos + tmp))
        pos = pos + tmp
    if data is not None:
        return [data[a:b] for a,b in res]
    return res