#!/usr/bin/env python3


import argparse
import os
import json
from pycocotools.coco import COCO
import numpy as np
import copy


def get_args():
    parser = argparse.ArgumentParser(description="Finds the length of a dataset")
    parser.add_argument('input_coco', help="Path to the coco format dataset. Order the dataset before using it here. Also filter for bad images (without pose)")
    args = parser.parse_args()
    return args


def calc_dist(a,b):
    dist = np.sqrt(np.sum((a - b)**2))
    return dist

def main(args):
    coco = COCO(args.input_coco)
    all_image_ids = coco.getImgIds()

    acc_dist = 0.0

    last_position = np.array(coco.loadImgs(ids=[all_image_ids[0]])[0]['pose']['position'][:2])

    for image in coco.loadImgs(all_image_ids):
        new_position = np.array(image['pose']['position'][:2])
        dist = calc_dist(new_position, last_position)  # Calc distance between previous positon
        last_position = new_position
        acc_dist += dist
        print(dist)

    print("Distance: ", acc_dist, "m")
    print("Distance: ", acc_dist/1000., "km")

    print("Mean distance between images: ", acc_dist/float(len(all_image_ids)))

if __name__ == "__main__":
    main(get_args())
