import sys
mylibs = ["/data2/gits/pyhej"]
for mylib in mylibs:
    if mylib not in sys.path:
        sys.path.insert(0, mylib)


import os
import copy
import json
import codecs
import cv2 as cv
from pyhej.utils import set_dir


def correct_bbox(x, x0, size):
    x -= x0
    if x < 0:
        return 0
    elif x > size:
        return size
    return x


# VGG Image Annotator saves each image in the form:
# { 'filename': '28503151_5b5b7ec140_b.jpg',
#   'regions': [
#     {
#       'region_attributes': {},
#       'shape_attributes': {
#         'all_points_x': [...],
#         'all_points_y': [...],
#         'name': 'polygon'}
#     },
#     ... more regions ...
#   ],
#   'size': 100202
# }
def regions_filter(regions, x0, y0, size):
    sub_regions = []
    for region in regions:
        region = copy.deepcopy(region)
        region_attributes = region["region_attributes"]
        shape_attributes = region["shape_attributes"]

        if shape_attributes["name"] == "rect":
            x = shape_attributes["x"]
            y = shape_attributes["y"]
            w = shape_attributes["width"]
            h = shape_attributes["height"]
            shape_attributes = {"name": "polygon",
                                "all_points_x": [x,   x, x+w, x+w],
                                "all_points_y": [y, y+h, y+h,   y]}

        xs = [correct_bbox(i, x0, size) for i in shape_attributes["all_points_x"]]
        ys = [correct_bbox(i, y0, size) for i in shape_attributes["all_points_y"]]

        if max(xs)-min(xs)<10 or max(ys)-min(ys)<5:
            continue

        shape_attributes["all_points_x"] = xs
        shape_attributes["all_points_y"] = ys
        sub_regions.append({"region_attributes": region_attributes, "shape_attributes": shape_attributes})

    return sub_regions


root = "/data2/gits/note-deep-learning/others/ubd_dataset31"
output_dir = "/data2/gits/note-deep-learning/others/tmps/ubd_dataset31"
jsonfile = "via_region_data.json"
cutting_size = 512
cutting_cover = 32


dataset = {}
set_dir(output_dir, rm=True)


with codecs.open(os.path.join(root, jsonfile), "r", "utf-8") as reader:
    annotations = json.load(reader)


for annotation in annotations.values():
    filename = annotation["filename"]
    regions = annotation["regions"]

    image = cv.imread(os.path.join(root, filename))
    y_max, x_max = image.shape[:2]

    for x in range(0, x_max-cutting_size, cutting_size-cutting_cover):
        for y in range(0, y_max-cutting_size, cutting_size-cutting_cover):
            sub_filename = "{}_{}_{}.jpg".format(".".join(filename.split(".")[:-1]), x, y)
            sub_regions = regions_filter(regions, x, y, cutting_size)

            sub_image = image[y:y+cutting_size, x:x+cutting_size]
            cv.imwrite(os.path.join(output_dir, sub_filename), sub_image)

            dataset[sub_filename] = {"filename": sub_filename, "regions": sub_regions}


with codecs.open(os.path.join(output_dir, jsonfile), "w", "utf-8") as writer:
    writer.write(json.dumps(dataset, indent=4))


print("output_dir: {}\njsonfile: {}".format(output_dir, jsonfile))


"""
# view annotations @Jupyter
output_dir = "/data2/gits/note-deep-learning/mask_rcnn/tmps/ubd_dataset31"
jsonfile = "via_region_data.json"

import os
import json
import codecs
import random
from PIL import Image, ImageDraw

with codecs.open(os.path.join(output_dir, jsonfile), "r", "utf-8") as reader:
    annotations = json.load(reader)

ikey = random.choice(list(annotations.keys()))
filename = os.path.join(output_dir, annotations[ikey]["filename"])
image = Image.open(filename)
draw = ImageDraw.Draw(image)
for r in annotations[ikey]["regions"]:
    xs = r["shape_attributes"]["all_points_x"]
    ys = r["shape_attributes"]["all_points_y"]
    draw.polygon([(x, y) for x, y in zip(xs, ys)], outline=(255,0,0))
image
"""