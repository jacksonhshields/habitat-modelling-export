#!/usr/bin/env python3

import json
from pycocotools.coco import COCO
import copy
import numpy as np
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser(description="Balances a coco file")
    parser.add_argument('input_coco', type=str, help="Path to the input coco format dataset")
    parser.add_argument('output_coco', type=str, help="Path to the output coco format dataset")
    args = parser.parse_args()
    return args

def main(args):
    # Input the dataset
    coco = COCO(args.input_coco)
    cat_ids = coco.getCatIds()
    # Initialise the lists
    ann_ids = []
    img_ids = []
    cat_idx = {}
    # Initialise dictionaries
    cat_indexes = {}
    cat_counts = {}
    cat_max = {}
    # Create counts, index lists etc.
    for cid in cat_ids:
        # Indexes
        cat_idx[cid] = 0
        cat_indexes[cid] = coco.getAnnIds(catIds=[cid])
        random.shuffle(cat_indexes[cid])
        cat_counts[cid] = 0
        cat_max[cid] = len(coco.getAnnIds(catIds=[cid]))

    # Add to dataset until minimum value is reached
    while min(cat_counts.values()) < min(cat_max.values()):
        # Get the category id with the least count
        cid = list(cat_counts.keys())[np.argmin(list(cat_counts.values()))]
        # Get the next annotation id
        aid = cat_indexes[cid][cat_idx[cid]]
        # Update the idx counter
        cat_idx[cid] += 1
        # Check if this annotation has already been extracted
        if aid in ann_ids:
            continue
        # Get the annotation
        ann = coco.loadAnns(ids=[aid])[0]
        # Get all annotation ids associated with this image
        extra_ann_ids = coco.getAnnIds(ann['image_id'])
        # Add these annotations
        ann_ids.extend(extra_ann_ids)
        img_ids.append(ann['image_id'])
        for ann in coco.loadAnns(extra_ann_ids):
            cat_counts[ann['category_id']] += 1

    # Create the output datasets
    outds = copy.deepcopy(coco.dataset)
    outds['categories'] = coco.dataset['categories']
    outds['images'] = coco.loadImgs(ids=img_ids)
    outds['annotations'] = coco.loadAnns(ids=ann_ids)
    json.dump(outds, open(args.output_coco, 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    main(get_args())
