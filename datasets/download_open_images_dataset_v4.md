# Open Images Dataset V4
https://storage.googleapis.com/openimages/web/download.html

## Boxes(bbox.csv)
每行定义一个边界框:

- ImageID: 图像ID
- Source: `freeform`和`xclick`是手动绘制的框.`activemil`是使用方法生成的,经过人工验证,在`IoU>0.7`时是准确的
- LabelName: 此框所属的对象类的MID
- Confidence: 虚拟值,始终为1
- XMin, XMax, YMin, YMax: 框的坐标.XMin在`[0,1]`中,其中0是最左边的像素,1是图像中最右边的像素.Y坐标从顶部像素0到底部像素1
- IsOccluded: 表示对象被图像中的另一个对象遮挡
- IsTruncated: 表示对象超出图像边界
- IsGroupOf: 表示该框横跨一组对象(例如,一张鲜花或一群人)
- IsDepiction: 表示对象是描绘(例如,对象的卡通或绘图,而不是真实的物理实例)
- IsInside: 表示从对象内部拍摄的照片(例如,汽车内部或建筑物内部)

>属性取值:1表示存在,0表示不存在,-1表示未知.

## Image Labels(boxable.csv)
经过人工验证和机器生成的图像级标签:

- Source: `verification`是由Google内部注释人员验证的标签.`crowdsource-verification`是Crowdsource应用程序验证的标签.`machine`是机器生成的标签
- Confidence: 经过人工验证图像中存在的标签具有`confidence=1`(正标签).人工验证图像中缺少的标签具有`confidence=0`(负标签).机器生成的标签具有分数置信度,通常大于`0.5`

## Image IDs(rotation.csv)
它具有图像`URL,OpenImages ID,rotation information`,以及标题,作者和许可证信息.

- OriginalSize: 原始图像的下载大小
- OriginalMD5: base64编码二进制MD5
- Rotation: 图像应逆时针转动的角度,以匹配用户预期的方向

## Class Names
通过查看`class-descriptions*.csv`,可以将MID格式的类名转换为其简短描述.
