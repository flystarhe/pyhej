## AI医疗
- [生物医学图像分析中的重大挑战](https://grand-challenge.org/All_Challenges)
- github https://github.com/sfikas/medical-imaging-datasets
- github https://github.com/beamandrew/medical-data

## DRIVE: Digital Retinal Images for Vessel Extraction
>用于血管提取的数字视网膜图像

DRIVE数据库已经建立起来,能够比较研究视网膜图像中血管的分割.邀请研究团队在这个数据库上测试他们的算法,并通过这个[网站](https://www.isi.uu.nl/Research/Databases/DRIVE/index.html)与其他研究人员分享结果.在这个[页面](https://www.isi.uu.nl/Research/Databases/DRIVE/index.html)上,可以找到关于下载数据库和上传结果的说明,并且可以检查各种方法的结果.

unzip to datasets dir:
```
#unzip DRIVE.zip -d /data2/object-masking/datasets/
#tree -L 2 /data2/object-masking/datasets/DRIVE/
/data2/object-masking/datasets/DRIVE/
├── test
│   ├── 1st_manual
│   ├── 2nd_manual
│   ├── images
│   └── mask
└── training
    ├── 1st_manual
    ├── images
    └── mask
```

### Image Sciences Institute
公开可用的注释图像数据库有利于比较研究.在这个页面上,我们提供了公共图像数据库的链接.这些数据库已经由Image Sciences Institute提供,或者在我们的支持下构建.

- AMIDA13: 乳腺癌组织学图像有丝分裂检测中的挑战
- DRIVE: 用于血管提取的数字视网膜图像
- EMPIRE10: 肺部CT扫描的挑战
- PROSTATE: MR前列腺图像数据库
- SCR: 胸部X光片的分割
- SLIVER07: 腹部CT扫描中肝脏的分割

## Carvana Image Masking Challenge
在本次比赛中,您将面临挑战,要开发一种自动删除照相馆背景的算法.这将允许Carvana在各种背景上叠加汽车.您将分析照片的数据集,涵盖各种年份,制作和模型组合的不同车辆.[url](https://www.kaggle.com/c/carvana-image-masking-challenge)

### Data
该数据集包含大量汽车图像(如`.jpg`文件).每辆车都有16个图像,每一个都以不同的角度拍摄.每辆车都有一个唯一的ID,图像根据`id_01.jpg`,`id_02.jpg`命名.除了图像之外,还提供了关于汽车制造,型号,年份和修剪的一些基本元数据.[url](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

```
/train
train_masks.csv
```

## Open-i

### Chest X-ray images
download here [NLMCXR_png.tgz](https://openi.nlm.nih.gov/faq.php):

```
#mkdir /data2/datasets/open-i
#tar -zxf NLMCXR_png.tgz -C /data2/datasets/open-i
#tree -L 2 /data2/datasets/open-i
```

### Montgomery County X-ray Set
download here [NLM-MontgomeryCXRSet.zip](https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip):

```
#unzip NLM-MontgomeryCXRSet.zip -d /data2/object-masking/datasets/tmps/
#tree -L 2 /data2/object-masking/datasets/tmps/
/data2/object-masking/datasets/tmps/
├── __MACOSX
│   └── MontgomerySet
└── MontgomerySet
    ├── ClinicalReadings
    ├── CXR_png
    ├── ManualMask
    └── NLM-MontgomeryCXRSet-ReadMe.pdf
```
