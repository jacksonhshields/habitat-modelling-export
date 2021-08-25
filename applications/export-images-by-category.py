#!/usr/bin/env python3


import argparse
from pycocotools.coco import COCO
import numpy as np
import os
import glob
from PIL import Image
import logging
import json
import csv
import utm
import datetime


def get_args():
    parser = argparse.ArgumentParser(description="Orders a COCO dive dataset so the images appear in order of collection")
    parser.add_argument('coco_path', help="Path to the coco path dataset")
    parser.add_argument('output_dir', help="Path to the output coco dataset")
    parser.add_argument('--resize', type=str, help="Resize the images during export. Use 0.5 to export images half the size. Use 256x256 to export images to a specific size")
    args = parser.parse_args()
    return args


def main(args):
    coco = COCO(args.coco_path)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resize:
        use_resize = True
        if 'x' in args.resize:
            new_size = tuple([int(x) for x in args.resize.split('x')])
            resize_ratio = None
        else:
            resize_ratio = float(args.resize)
    else:
        use_resize = False

    for cat in coco.loadCats(coco.getCatIds()):
        # Make the output directory
        cat_dir = os.path.join(args.output_dir, cat['name'])
        os.makedirs(cat_dir, exist_ok=True)
        for image in coco.loadImgs(coco.getImgIds(catIds=[cat['id']])):
            # Load the image
            pimg = Image.open(image['path'])
            if resize_ratio:
                new_size = (int(pimg.size[0]*resize_ratio), int(pimg.size[1]*resize_ratio))
            pimg = pimg.resize(new_size)
            pimg.save(os.path.join(cat_dir, os.path.basename(image['path'])))

if __name__ == "__main__":
    main(get_args())

