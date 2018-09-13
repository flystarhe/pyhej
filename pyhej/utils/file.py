import os
import re
import shutil
import requests
from io import BytesIO
from urllib.request import urlretrieve
from pyhej.utils import set_dir, set_parent


URL_REGEX = re.compile(r"http://|https://|ftp://")


def read(filename, default=None):
    try:
        if URL_REGEX.match(filename):
            return BytesIO(requests.get(filename).content)
        with open(filename, "rb") as f:
            return f.read()
    except:
        return default


def download_file(furl, filename="000", format=None, clever=False):
    set_parent(filename, "777", rm=False)
    res = urlretrieve(furl, filename)
    if clever:
        format = res[1].get_content_subtype()
    if format is not None:
        shutil.move(res[0], res[0]+"."+format)


def download_files(furls, outdir="tmps", prefix="", suffix="", rm=False):
    set_dir(outdir, "777", rm=rm)
    for i, furl in enumerate(furls):
        fpath = "{}{:06d}.{}".format(prefix, i, suffix)
        fpath = os.path.join(outdir, fpath)
        try:
            urlretrieve(furl, fpath)
        except Exception:
            print("* Fail | [{}, {}]".format(i, furl))