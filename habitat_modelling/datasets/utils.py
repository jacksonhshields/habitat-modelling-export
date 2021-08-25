#!/usr/bin/env python3

import numpy as np
from pycocotools.coco import COCO
import warnings

def calculate_class_weights(dataset, min_count=None, min_ratio=0.01, num_classes=None):
    """

    NOTE: Assumes there is only one annotation per image

    Args:
        dataset:
        min_clip: (int)

    Returns:

    """
    # Calculate counts
    counts = {cat['id']:0 for cat in dataset['categories']}
    for ann in dataset['annotations']:
        counts[ann['category_id']] += 1

    # Calculate class weights
    if min_count:
        min_clip = min_count
    else:
        min_clip = min_ratio*max(counts.values())

    if num_classes is None:
        num_classes = len(dataset['categories'])

    count_arr = np.zeros([num_classes])
    for k,v in counts.items():
        count_arr[k] = v

    minval = np.min(count_arr[np.nonzero(count_arr)])
    if minval < min_clip:
        minval = min_clip
    count_arr[count_arr == 0] = 1
    weights = minval * 1/(count_arr)
    return weights







