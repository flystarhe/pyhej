# None
- https://www.dicomstandard.org/current/
- https://dicom.innolitics.com/ciods
- (0020, 000e) Series Instance UID
- (0008, 0018) SOP Instance UID

## Dicom (Pixel Data)
显示医用灰度图像的相关说明.

### (0028, 0002) Samples per Pixel                   US: 1
每一个像素的取样数,一般来说CT/MR/DR等灰度图像都是1,而彩超等彩色图像都是3,分别表示`R,G,B`.

### (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
我们经常碰到的有以下几种类型:

- Monochrome2:一般的灰度图像都采用这种,Pixel值越大,图像就越白
- Monochrome1:只有部分CR/DR图像使用,Pixel值越大,图像就越黑
- Palette Colour:一般用于彩超图像,每个像素占用8位或者16位,调色板保存在`[0028,1201]`,`[0028,1202]`和`[0028,1203]`的属性中
- RGB:这是最常用的彩色图像格式

### (0028, 0030) Pixel Spacing                       DS: ['0.414062', '0.414062']
图像像素间距,读取`Pixel Data`的时候不需要,主要用于长度测量.

### (0028, 0100) Bits Allocated                      US: 16
一个像素取样点存储时分配到的位数.对于灰度图像,如果是256级灰阶,一般就是8位.如果高于256级灰阶,一般就采用16位.

### (0028, 0101) Bits Stored                         US: 16
一个像素取样点存储时使用到的位数.比方说示例中CT影像,采用的是4K灰阶,像素值取值范围为`[0-4095]`,所以使用到的位数为12位.

### (0028, 0102) High Bit                            US: 15
最高位序号,它定义了存储点在分配的内存中的排列方式,它的值是最后一个bit的序号.如果第一个bit放在0位,那么最后一个bit为`Bits Stored - 1`.

### (0028, 0103) Pixel Representation                US: 1
如果这个值为0,这表明是无符号类型,其VR类型应该为US(Unsigned Short).如果这个值为1,这表明为有符号类型,其VR类型应该为SS(Signed Short).

### (0028, 0120) Pixel Padding Value                 SS: -2000
图像中具有此值的任何像素都不被视为有意义的对象,而是作为背景颜色.

### (0028, 1052) Rescale Intercept                   DS: "-1024"
用于根据像素值计算原始值,比方说,CT可以用于计算HU值.比方说:
```
HU = Rescale Slope * X + Rescale Intercept
```

### (0028, 1053) Rescale Slope                       DS: "1"
用于根据像素值计算原始值,同`Rescale Intercept`.

### (0028, 1054) Rescale Type                        LO: 'HU'
重新缩放类型.