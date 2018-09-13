# COCO 2017

## install [pycocotools](https://github.com/cocodataset/cocoapi)
```
git clone https://github.com/cocodataset/cocoapi.git tmp_cocoapi
cd tmp_cocoapi/PythonAPI
make
python setup.py install
cd ../..
rm -rf tmp_cocoapi
```

## download [COCO 2017](http://cocodataset.org/)
If using `wget`:
```
cd /data2/datasets
mkdir -p COCO && cd COCO

wget http://images.cocodataset.org/zips/train2017.zip && \
wget http://images.cocodataset.org/zips/val2017.zip && \
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q *.zip
rm *.zip
```

If using `aria2c`:
```
cd /data2/datasets
mkdir -p COCO && cd COCO

aria2c -x 10 -j 10 http://images.cocodataset.org/zips/train2017.zip && \
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/val2017.zip && \
aria2c -x 10 -j 10 http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q *.zip
rm *.zip
```

## instances*
```
instances_*.json:
    key: annotations
        list[0]:
            'bbox': [473.07, 395.93, 38.65, 28.67],
            'category_id': 18,
            'id': 1768,
            'image_id': 289343,
            'iscrowd': 0,
            'segmentation': list or dict

    key: categories
        list[0]:
            'id': 1
            'name': 'person'
            'supercategory': 'person'

    key: images
        list[0]:
            'id': 397133,
            'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
            'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
            'file_name': '000000397133.jpg',
            'height': 427,
            'width': 640
```

## usage
```python
from pycocotools import mask as maskUtils

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

root = "/data2/datasets/COCO/"
subset = "val2017"

from pycocotools.coco import COCO

image_dir = "{}/{}".format(root, subset)
coco = COCO("{}/annotations/instances_{}.json".format(root, subset))
class_ids = sorted(coco.getCatIds())
#image_ids = list(coco.imgs.keys())
image_ids = []
for id in class_ids:
    image_ids.extend(list(coco.getImgIds(catIds=[id])))
# Remove duplicates
image_ids = list(set(image_ids))

image_id = image_ids[0]
annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None))

# masks: A bool array of shape [height, width, instance count] with a binary mask per instance.
#   TensorFlow/Keras: channels_last
#   PyTorch: channels_first
masks = []
class_ids = []
for annotation in annotations:
    m = annToMask(annotation, coco.imgs[image_id]["height"], coco.imgs[image_id]["width"])
    class_id = annotation['category_id']
    if m.max() < 1:
        # Some objects are so small. Skip those objects.
        continue
    masks.append(m)
    class_ids.append(class_id)
if len(class_ids):
    masks = np.stack(masks, axis=2).astype(np.bool)
    class_ids = np.array(class_ids, dtype=np.int32)
#np.allclose(masks, masks)
```
