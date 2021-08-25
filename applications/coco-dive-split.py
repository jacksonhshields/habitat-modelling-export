#!/usr/bin/env python3


import argparse
import os
import json
from pycocotools.coco import COCO
import numpy as np
import copy
import warnings

def get_args():
    parser = argparse.ArgumentParser(description="Splits a single dive into training and validation sections, and avoids any overlap between bathymetry patches")
    parser.add_argument('input_coco', help="Path to the coco format dataset. Order the dataset before using it here. Also filter for bad images (without pose)")
    parser.add_argument('output_dir', help="Path to the output directory. Output coco datasets will be named input_coco_basename.train.json, input_coco_basename.val.json")
    parser.add_argument('--patch-size', type=float, help="The patch size in meters")
    parser.add_argument('--partitions', type=str, help="The partitions to extract, e.g. train,val", required=True)
    parser.add_argument('--distances', type=str, help="The number of meters in each partition segment, comma separated")
    args = parser.parse_args()
    return args


def calc_dist(a,b):
    dist = np.sqrt(np.sum((a - b)**2))
    return dist


def guess_dists(coco, patch_size, num_partitions):
    all_image_ids = coco.getImgIds()
    last_position = np.array(coco.loadImgs(ids=[all_image_ids[0]])[0]['pose']['position'][:2])
    acc_dist = 0.0
    for image in coco.loadImgs(ids=all_image_ids):
        new_position =np.array(image['pose']['position'][:2])
        dist = calc_dist(new_position, last_position)  # Calc distance between previous positon
        acc_dist += dist  # Add to distance count
        last_position = new_position  # Set new position

    leftover_dist = acc_dist - patch_size*num_partitions*2.0  # The distance for the segments is the total distance minus the number of partitions with a gap on each side
    # leftover_dist = acc_dist
    dist_per_partition = leftover_dist / num_partitions
    return [dist_per_partition for _ in range(num_partitions)]

def main(args):
    coco = COCO(args.input_coco)
    all_image_ids = coco.getImgIds()

    partitions = {}
    if args.distances:
        distances_for_splits = [float(x) for x in args.distances.split(',')]
    else:
        distances_for_splits = guess_dists(coco, args.patch_size, len(args.partitions.split(',')))
    for part, dist in zip(args.partitions.split(','), distances_for_splits):
        partitions[part] = {
            "image_ids": [],
            "distance": float(dist)
        }
    part_idx = 0  # Stores the
    current_part = list(partitions.keys())[part_idx]

    gap_section = False  # Gap sections are inbetween segments where there cannot be overlapping

    acc_dist = 0.0
    last_position = np.array(coco.loadImgs(ids=[all_image_ids[0]])[0]['pose']['position'][:2])

    for image in coco.loadImgs(ids=all_image_ids):
        new_position =np.array(image['pose']['position'][:2])
        if gap_section:
            # dist = calc_dist(new_position, last_position) =
            if np.all(np.abs(new_position - last_position) > args.patch_size):  # Check if still overlapping with previous patch
                gap_section = False  # Out of range of previous patch
                last_position = new_position
            else:
                continue
        else:
            dist = calc_dist(new_position, last_position)  # Calc distance between previous positon
            acc_dist += dist  # Add to distance count
            last_position = new_position  # Set new position
            partitions[current_part]["image_ids"].append(image['id'])  # Add image to this partition

            if acc_dist > partitions[current_part]["distance"]:  # Check if the accumulated distance is greater than the max
                part_idx += 1  # Increment the partition idx
                if part_idx >= len(partitions.keys()):  # If overflow, reset to zero
                    part_idx = 0
                    print("Overflow of partitions!")
                current_part = list(partitions.keys())[part_idx]  # Set the new current part
                gap_section = True  # Make it a gap section
                acc_dist = 0.0

    for p, part in partitions.items():  # Output new coco files
        # Create output coco dataset
        outcoco = copy.deepcopy(coco.dataset)
        if len(part['image_ids']) == 0:
            warnings.warn("%s has no images" %p)
            outcoco['images'] = []
            outcoco['annotations'] = []
        else:
            outcoco['images'] = coco.loadImgs(part['image_ids'])
            outcoco['annotations'] = coco.loadAnns(coco.getAnnIds(imgIds=part['image_ids']))
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_coco))[0] + '.' + p + '.json')
        json.dump(outcoco, open(output_path, 'w'), sort_keys=True, indent=4)

if __name__ == "__main__":
    main(get_args())
