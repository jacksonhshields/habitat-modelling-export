#!/usr/bin/env python3

import argparse
import simplekml
import numpy as np
import geojson
from pycocotools.coco import COCO
import webcolors
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
import pymap3d
import pandas as pd
import shapely
import shapely.geometry
import copy
import json

def get_args():
    parser = argparse.ArgumentParser(description="Ensures all coco paths have image/w")
    parser.add_argument('input_coco', help="The dataset that is to be filtered. A new copy of this dataset will have images removed around these areas.")
    parser.add_argument('output_coco', help="Path to the output coco file")
    args = parser.parse_args()
    return args

def main(args):
    input_coco = COCO(args.input_coco)
    id_list = []
    for image in input_coco.loadImgs(input_coco.getImgIds()):
        if 'geo_location' in image:
            id_list.append(image['id'])


    # Create output dataset
    output_ds = copy.deepcopy(input_coco.dataset)
    output_ds['images'] = input_coco.loadImgs(ids=id_list)
    output_ds['annotations'] = input_coco.loadAnns(input_coco.getAnnIds(imgIds=id_list))
    # keep categories the same


    json.dump(output_ds, open(args.output_coco, 'w'), indent=4, sort_keys=True)

if __name__ == "__main__":
    main(get_args())
