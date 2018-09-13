import os
from pathlib import Path


def pyhej_data_path():
    root = Path(__file__).parent.parent
    return root.as_posix()


if "PYHEJ_DATA" not in os.environ:
    os.environ["PYHEJ_DATA"] = pyhej_data_path()