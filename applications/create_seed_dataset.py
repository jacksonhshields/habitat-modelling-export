#!/usr/bin/env python3
import argparse
import os
import json
import copy
from pycocotools.coco import COCO
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Create a seed coco dataset from an existing dataset")
    parser.add_argument('--input', '-i', help="The input coco file", required=True)
    parser.add_argument('--output', '-o', help="The input coco file", required=True)
    parser.add_argument('--ratio', type=float, help="The ratio of images to use in the seed coco. Can't be used with --num-samples")
    parser.add_argument('--num-samples', type=int, help="The number of images to be used in the seed coco. Can't be used with --ratio")
    args = parser.parse_args()
    if args.ratio is not None and args.num_samples is not None:
        raise ValueError("Specify either --ratio or --num-samples")
    if args.ratio is not None and (args.ratio > 1.0 or args.ratio < 0.0):
        raise ValueError("--ratio should be between 0 and 1")
    if args.ratio is None and args.num_samples is None:
        raise ValueError("Either or --ratio or --num-samples needs to be given")
    return args


def main(args):
    coco = COCO(args.input)
    output = copy.deepcopy(coco.dataset)

    all_iids = coco.getImgIds()
    all_iids = np.asarray(all_iids)
    total_samples = len(all_iids)
    if args.ratio is not None:
        num_samples = int(round(total_samples * args.ratio))
    else:
        num_samples = args.num_samples
    print("Selecting %d/%d samples from the coco dataset" % (num_samples, total_samples))

    np.random.shuffle(all_iids)  # shuffles in place
    selected_iids = all_iids[:num_samples]
    selected_iids = selected_iids.tolist()

    images = coco.loadImgs(ids=selected_iids)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=selected_iids))

    output['images'] = images
    output['annotations'] = annotations

    json.dump(output, open(args.output, 'w'))

if __name__ == "__main__":
    main(get_args())