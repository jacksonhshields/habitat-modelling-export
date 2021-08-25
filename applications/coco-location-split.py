#!/usr/bin/env python3


import argparse
import json
from pycocotools.coco import COCO
import ast
import shapely
import shapely.geometry
import copy
import warnings

def get_args():
    parser = argparse.ArgumentParser(description="Filters a COCO file based on geographic location")
    parser.add_argument('--input-coco', '-i', help="The input coco dataset")
    parser.add_argument('--boundary', '-b', help="The boundary polygon that contains the points. \
    Format= '[[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]' . If n==2, it will be a rectangle bounding box, else it will be a polygon")
    parser.add_argument('--output-coco', '-o', help="Path to the output coco file")
    args = parser.parse_args()
    return args


def main(args):
    coco = COCO(args.input_coco)
    corners = ast.literal_eval(args.boundary)
    if len(corners) == 2:
        boundary = shapely.geometry.box(*corners)
    else:
        boundary = shapely.geometry.Polygon(corners)

    image_list = []
    ann_list = []
    for image in coco.loadImgs(coco.getImgIds()):
        if 'geo_location' not in image:
            warnings.warn("geo_location not in image")
            continue

        geo_loc = image['geo_location'][:2]
        point = shapely.geometry.Point(geo_loc)
        if boundary.contains(point):
            image_list.append(image)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
            if len(anns) > 0:
                ann_list.extend(anns)

    outcoco = copy.deepcopy(coco.dataset)
    outcoco['images'] = image_list
    outcoco['annotations'] = ann_list

    json.dump(outcoco,open(args.output_coco, 'w'), indent=4)

if __name__ == "__main__":
    main(get_args())