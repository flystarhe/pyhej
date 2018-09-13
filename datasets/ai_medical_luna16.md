# Lung nodule analysis 2016
- https://luna16.grand-challenge.org/download/
- `flystar5i/Gfglzyu90`

## 数据
对于这一挑战,我们使用公开的[LIDC/IDRI数据库](http://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).此数据使用知识共享署名3.0 Unported许可.我们排除了切片厚度大于`2.5mm`的扫描.总共包括888次CT扫描.LIDC/IDRI数据库还包含使用4位经验丰富的放射科医师在两阶段注释过程中收集的注释.每位放射科医师都将病灶标记为非结节/结节`<3mm`/结节`>=3mm`.我们挑战的参考标准包括`>=3mm`的所有结节,至少有4名放射科医师接受了这些结节.未包括在参考标准中的注释(非结节/结节`<3mm`/仅由1或2名放射科医师注释的结节)被称为无关的发现.[评估脚本包](https://www.dropbox.com/s/wue67fg9bk5xdxt/evaluationScript.zip?dl=0)`annotations_excluded.csv`中提供了不相关的发现列表.

数据在下载页面上提供.数据结构如下:

- `subset0.zip..subset9.zip`:包含所有CT图像的10个zip文件
- `annotations.csv`:包含用作`结节检测`轨道参考标准的注释的csv文件
- `sampleSubmission.csv`:正确格式的提交文件示例
- `candidates_V2.csv`:包含`误报减少`轨道的候选位置的csv文件

其他数据包括:

- 评估脚本:LUNA16框架中使用的评估脚本
- 肺部分割:包含使用自动算法计算的CT图像的肺部分割的目录
- `additional_annotations.csv`:包含我们观察者研究中的其他结节注释的csv文件.即将推出

>注意:数据集用于训练和测试数据集.为了便于重现,请使用给定的子集来训练算法进行10次交叉验证.

## 下载
为了下载这些文件,需要一个torrent客户端.Ubuntu推荐使用`Transmission`.

## mhd raw
python3:
```python
#https://github.com/shartoo/luna16_multi_size_3dcnn/blob/master/data_prepare.py
import numpy as np
import SimpleITK as sitk

mhd_file = "/data2/datasets/ai_medical_luna16/zips/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
itk_img = sitk.ReadImage(mhd_file)
origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x
img_array = img_array.transpose(2,1,0)      # transfer to x,y,z
```

LUNA2016 data prepare ,first step:
```python
#truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 400
    image_array[image_array <-1000] = -1000
```

LUNA2016 data prepare ,second step:
```python
#normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array.astype("float32")
```
