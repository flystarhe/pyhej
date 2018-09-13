# Pascal VOC
http://host.robots.ox.ac.uk/pascal/VOC/index.html

## download
If using `wget`:
```
cd /data2/datasets
mkdir -p VOC && cd VOC

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && \
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && \
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xf *.tar
rm *.tar
```

If using `aria2c`:
```
cd /data2/datasets
mkdir -p VOC && cd VOC

aria2c -x 10 -j 10 http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && \
aria2c -x 10 -j 10 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && \
aria2c -x 10 -j 10 http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xf *.tar
rm *.tar
```
