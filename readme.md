# PyHej

## requirements.txt
```
conda update conda
conda install python=3.5
conda install numpy=1.13.3
conda install pytorch torchvision -c pytorch
pip install --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.8.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.1.5
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install imgaug
pip install Augmentor
pip install pyannote.audio
pip install pyannote.metrics
```

## datasets
设置环境变量`PYHEJ_DATA`指向`datasets`的父目录.或在程序中指定:
```python
import os
if "PYHEJ_DATA" not in os.environ:
    os.environ["PYHEJ_DATA"] = "your/path/datasets_parent"
```

另:`os.getcwd()`查看当前工作目录,`os.chdir("new/work/dir")`切换工作目录.

## Module: gen_plate
车牌生成器,参考[szad670401](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)实现,对代码进行了精简,支持py3.

## Import
```python
import sys
mylibs = ["/data2/gits/pyhej"]
for mylib in mylibs:
    if mylib not in sys.path:
        sys.path.insert(0, mylib)
```
