# CST-Dataset
https://github.com/GeeshangXu/cst-dataset

CST-Dataset, Circle-Square-Triangle Dataset, is a simple small-scale object detection and segmentation dataset, has 1000 images, contains only circles, squares and triangles in different scales, and the file size is just 25MB.

考虑到一些重量级数据如PASCAL VOC或COCO,训练时间通常为数小时甚至数天.因此,十分钟的训练数据集可以作为初学者熟悉物体检测和分割的良好开端,或作为实施者的健全性检查数据集以快速检查模型实施的正确性.这个数据集是随机生成的,并有一些控制,以确保这些对象不会彼此重叠或与边界重叠.

```
image_id    object_id   class   bbox    mask_path
cst-600 cst-600-0   t   301 123 368 183 cst-600/masks/cst-600-0.jpg
cst-600 cst-600-1   s   300 270 335 306 cst-600/masks/cst-600-1.jpg
cst-600 cst-600-2   c   377 58 433 114  cst-600/masks/cst-600-2.jpg
cst-600 cst-600-3   s   334 374 397 437 cst-600/masks/cst-600-3.jpg
```

## Dataset download link
[CST-Dataset-1.0](https://github.com/GeeshangXu/cst-dataset/releases).

## Code to generate dataset
Because of fixed random seed, the result of generating will be the same with release file.

```
python3 generate_cst.py ./output_dir
```
