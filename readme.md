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
if "PYHEJ_DATA" not in os.environ:
    os.environ["PYHEJ_DATA"] = "your/path/datasets_parent"
```

另:`os.getcwd()`查看当前工作目录,`os.chdir("new/work/dir")`切换工作目录.

## Module: gen_plate
车牌生成器,参考[szad670401](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)实现,对代码进行了精简,支持py3.

## Git Guide
建议作为子模块引入到你的项目:
```
git submodule add https://github.com/flystarhe/pyhej.git module_pyhej
```

注意,克隆使用子模块的项目,执行`git submodule *`是必要的,否则子模块不可用.比如:
```
git clone https://github.com/flystarhe/proj_name.git && cd proj_name
git submodule init
git submodule update
```

技巧!!!当需要`pull`子模块时,如果你不想在子目录中手动抓取与合并,那么还有种更容易的方式:`git submodule update --remote`,Git将会进入子模块然后抓取并更新.在`push`前执行`git submodule update --remote`更是个好习惯.

如果需要移除子模块,请如下操作:
```
git rm --cached module_pyhej
rm -rf module_pyhej
```

对比版本:
```
git fetch origin
git diff origin/master master
```

回滚版本:
```
git log -5  #显示最近提交
git reset --hard xxx  #回滚到指定版本
```

`remote`管理:
```
git remote add origin https://github.com/flystarhe/proj_name_1.git
git remote set-url origin https://github.com/flystarhe/proj_name_2.git
```

## Import
```python
import sys
mylibs = ["/data2/gits/pyhej"]
for mylib in mylibs:
    if mylib not in sys.path:
        sys.path.insert(0, mylib)
```
