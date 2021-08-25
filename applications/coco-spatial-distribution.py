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
    parser = argparse.ArgumentParser(description="Adds the spatial categorical distribution of classes to each annotation")
    parser.add_argument('input_coco', help="The input coco dataset")
    parser.add_argument('output_coco', help="The output coco dataset")
    parser.add_argument('--mode', default='nearest', help="The mode to use, options are {nearest, interesections}")
    parser.add_argument('--exclusion-dist', type=float, default=20., help="The distance from the intersection points at which tp exclude data points from target coco")
    parser.add_argument('--k-each-side', type=int, default=40, help="The number of images to search either side of the target image")
    parser.add_argument('--self-intersections', action="store_true", help="Whether to use the self intersections to calculate neighbours")
    args = parser.parse_args()
    return args

def calc_distance(a,b):
    """
    Calculates the distance between two np arrays
    """
    return np.sqrt(np.sum((a-b)**2))

def calc_in_box(a, box_centre, xdist_max, ydistmax):
    """
    Calculates if the
    Args:
        a: the target point
        box_centre: the centre of the box
        xdist_max: Half width of the box
        ydistmax: Half height of the box

    Returns:

    """
    diff = np.abs(a - box_centre)
    if diff[0] < xdist_max and diff[1] < ydistmax:
        in_box = True
    else:
        in_box = False
    return in_box


def get_local_poses(coco):
    pose_list = []
    for n, image in enumerate(coco.loadImgs(coco.getImgIds())):
        pose = image['pose']['position'][:2]
        pose_list.append(pose)
    return pose_list

def get_self_intersections(coco, exclusion_dist):
    pose_list = get_local_poses(coco)
    reduced_pose_list = [pose_list[x] for x in range(0,len(pose_list),100)]
    mpoints = []
    for n in range(len(reduced_pose_list)-1):
        p = (reduced_pose_list[n], reduced_pose_list[n+1])
        mpoints.append(p)
    msr = shapely.geometry.MultiLineString(mpoints)

    intersection_points = []

    # Iterate through the line segments
    for i, seg1 in enumerate(msr):
        for j, seg2 in enumerate(msr):
            if seg1.intersects(seg2):
                intersection = np.array(seg1.intersection(seg1))
                intersection_points.append(intersection)
    img_ids = []
    # Get the pose ids near the intersection points
    for image in coco.loadImgs(coco.getImgIds()):
        img_pos = np.array(image['pose']['position'][:2])
        for intersection in intersection_points:
            if calc_distance(img_pos, intersection) > exclusion_dist:
                img_ids.append(image['id'])
    return img_ids


def intersections(args):
    # Read dataset
    coco = COCO(args.input_coco)

    all_img_ids = coco.getImgIds()

    num_classes = len(coco.getCatIds())

    neighbour_dict = {}

    new_ann_list = []

    for n, iid in enumerate(all_img_ids):
        image = coco.loadImgs(ids=[iid])[0]
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))[0]
        cat_id = ann['category_id']
        img_pose = np.array(image['pose']['position'][:2])
        min_idx = max(n - args.k_each_side, 0)
        max_idx = min(n + args.k_each_side, len(all_img_ids))
        neighbour_ids = all_img_ids[min_idx:max_idx]
        # neighbour_ids.remove(iid)  # TODO remove or not remove
        close_ids = []
        for image in coco.loadImgs(ids=neighbour_ids):
            neighbour_pose = np.array(image['pose']['position'][:2])
            if calc_distance(img_pose, neighbour_pose) < args.exclusion_dist:
                close_ids.append(image['id'])

        if args.self_intersections:
            close_ids.extend(get_self_intersections(coco, exclusion_dist=args.exclusion_dist))
            close_ids = list(set(close_ids))

        anns = coco.loadAnns(coco.getAnnIds(imgIds=close_ids))
        nearest_hot = np.zeros([num_classes])
        for ann in anns:
            nearest_hot[ann['category_id']] += 1
        nearest_hot_norm = nearest_hot / np.sum(nearest_hot)  # Normalize
        neighbour_dict[iid] = {
            'category_id': cat_id,
            'neighbours': nearest_hot_norm
        }
        # Create a new annotation which is duplicate of the current one, and add the distribution
        new_ann = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))[0]
        new_ann['neighbours'] = [int(x) for x in nearest_hot.tolist()]

        new_ann_list.append(new_ann)

    output_coco = copy.deepcopy(coco.dataset)
    output_coco['annotations'] = new_ann_list

    # Write the new coco file
    json.dump(output_coco, open(args.output_coco, 'w'), sort_keys=True, indent=4)

def nearest(args):
    # Read dataset
    coco = COCO(args.input_coco)

    all_img_ids = coco.getImgIds()

    new_ann_list = []
    num_classes = len(coco.getCatIds())


    for centre_image in coco.loadImgs(coco.getImgIds()):
        centre_anns = coco.loadAnns(coco.getAnnIds(imgIds=[centre_image['id']]))
        if len(centre_anns) == 0:
            continue
        box_centre = np.array(centre_image['pose']['position'][:2])
        nearest_hot = np.zeros([num_classes])
        for other_image in coco.loadImgs(coco.getImgIds()):
            other_pos = np.array(other_image['pose']['position'][:2])
            if calc_in_box(other_pos, box_centre, args.exclusion_dist, args.exclusion_dist):
                anns = coco.loadAnns(coco.getAnnIds(imgIds=other_image['id']))
                if len(anns) == 0:
                    continue
                ann = anns[0]
                other_cat = ann['category_id']
                nearest_hot[other_cat] += 1
        new_ann = coco.loadAnns(coco.getAnnIds(imgIds=[centre_image['id']]))[0]
        new_ann['neighbours'] = [int(x) for x in nearest_hot.tolist()]
        new_ann_list.append(new_ann)

    output_coco = copy.deepcopy(coco.dataset)
    output_coco['annotations'] = new_ann_list

    # Write the new coco file
    json.dump(output_coco, open(args.output_coco, 'w'), sort_keys=True, indent=4)








def main(args):
    if args.mode == "nearest":
        nearest(args)
    elif args.mode == "intersections":
        intersections(args)
    else:
        raise ValueError("Invalid mode, options are {nearest,intersections}")

if __name__ == "__main__":
    main(get_args())
