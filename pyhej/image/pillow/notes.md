# pillow

## python3之成像库pillow
http://www.cnblogs.com/zhangxinqi/p/9286506.html

```python
from PIL import Image

im = Image.open("*.jpg")
## im.size: (width, height)
## im.mode: "1", "L", "RGB", ..
## im.format: "jpg", "png", ..
## im.verify(): 检查文件,不解码图像数据
## im.copy(): 拷贝图像
## im.crop(box): 裁剪图像
## im.transpose(method): `PIL.Image.FLIP_LEFT_RIGHT/FLIP_TOP_BOTTOM/ROTATE_90`
## im.convert(mode): "L", "RGB", `L=R*299/1000+G*587/1000+B*114/1000`
## im.filter(Filter): Filter@`from PIL import ImageFilter`
## im.resize(size, filter=None): 返回图像的已调整大小的副本
## im.save(fp, format=None, **params): 将图像保存为指定的文件
## im.paste(im2, box): 粘贴`im2`到`im`
## im.getbbox(): 计算图像中非零区域的边界框
## im.getbands(): RGB图像上返回`("R","G","B")`
## im.thumbnail(size, resample=3): 修改图像为自身的缩略图
```
