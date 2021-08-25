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

def extract_cluster_labels(cluster_filepath):
    """
    Extracts the cluster labels from the cluster filepath.

    Args:
        cluster_filepath:

    Returns:

    """
    cluster_lookup = {} # key=image_fp, value=cluster label
    with open(cluster_filepath, 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter='\t')
        for row in data_reader:
            if row:
                if row[0][0] == '%':
                    continue
                # Remove the color/mono part and extension
                fp = '_'.join(row[2].split('_')[:-1])
                cluster_lookup[fp] = row[4]
    return cluster_lookup

def extract_gmc_cluster_labels(gmc_cluster_filepath):
    """

    Args:
        gmc_cluster_filepath:

    Returns:

    """
    cluster_lookup = {}
    with open(gmc_cluster_filepath, 'r') as data_file:
        data_reader = csv.reader(data_file, delimiter=' ')
        for row in data_reader:
            if row:
                if row[0][0] == '%':
                    continue
                # Remove the color/mono part and extension
                fp = '_'.join(row[0].split('_')[:-1])
                cluster_lookup[fp] = row[1]
    return cluster_lookup



def add_clusters_to_dataset(dataset, cluster_filepath, gmc=False):
    """
    Adds image list to the
    Args:
        dataset:
        cluster_filepath:

    Returns:

    """
    if gmc:
        cluster_lookup = extract_gmc_cluster_labels(cluster_filepath)
    else:
        cluster_lookup = extract_cluster_labels(cluster_filepath)

    uniq_cats = sorted(list(set(cluster_lookup.values())))
    categories = []
    for i,cl in enumerate(uniq_cats):
        cat = {
            'id': int(cl),
            'name': str(cl),
            'supercategory': ""
        }
        categories.append(cat)

    annotations = []
    ann_count = 1  # Index from 1
    for image in dataset['images']:
        reduced_fn = '_'.join(image['file_name'].split('_')[:-1])
        if reduced_fn in cluster_lookup:
            catID = cluster_lookup[reduced_fn]  # Label
            ann = {
                'id': ann_count,
                'category_id': int(catID),
                'image_id': image['id'],
                "annotation_type": "point",  # Put it as a point label so it will show up in label software
                'bbox': [1, 2, 3, 4], # Needs to be there
                "iscrowd": 0,
                "occluded": False,
                "segmentation": [[1, 2, 3, 4]],
                'area': 10.0
            }
            ann_count += 1
            annotations.append(ann)
    dataset['annotations'] = annotations
    dataset['categories'] = categories
    return dataset

def get_args():
    parser = argparse.ArgumentParser(description="Adds cluster labels from either .csv or .data to the coco dataset")
    parser.add_argument('coco_path', help="Path to the acfr_marine dive processed directory")
    parser.add_argument('output_coco', help="Output path for the coco json")
    parser.add_argument('--cluster-data-file', help="The image label file generated from the clustering process. Usually called image_labels.data, in the renav directory")
    parser.add_argument('--cluster-gmc-file', help="The cluster file developed from clustering on all of Tasmania using the GMC process")
    args = parser.parse_args()
    return args

def main(args):
    if args.cluster_data_file and args.cluster_gmc_file:
        raise ValueError("Both --cluster-data-file and --cluster-gmc-file cannot be given")
    elif args.cluster_data_file is None and args.cluster_gmc_file is None:
        raise ValueError("Either cluster file needs to be given")
    elif args.cluster_data_file:
        cluster_file = args.cluster_data_file
        gmc = False
    elif args.cluster_gmc_file:
        cluster_file = args.cluster_gmc_file
        gmc = True
    else:
        raise ValueError("invalid arguments")

    dataset = json.load(open(args.coco_path, 'r'))
    dataset = add_clusters_to_dataset(dataset, cluster_file, gmc)

    os.makedirs(os.path.dirname(args.output_coco), exist_ok=True)

    json.dump(dataset, open(args.output_coco, 'w'), indent=4, sort_keys=True)

if __name__ == "__main__":
    main(get_args())