# ImageNet dataset
ImageNet大规模视觉识别挑战(ILSVRC)数据集有1000个类别和120万个图像.图像不需要预处理或打包在任何数据库中,但需要将验证图像移动到适当的子文件夹中.

## 1 下载图片
http://image-net.org/download-images

## 2 提取数据
提取训练数据:
```bash
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

提取验证数据:
```bash
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
