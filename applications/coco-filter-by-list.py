#!/usr/bin/env python3

import json
from pycocotools.coco import COCO
import argparse
import sys
import os
from contextlib import redirect_stdout
import csv
import copy


def say(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, file=sys.stderr, **kwargs)


def die(*args, **kwargs):
    say("error: ", *args, verbose=True, **kwargs)
    sys.exit(1)


class Verbose:
    @staticmethod
    def write(line):
        line = line.strip()
        if line:
            say(line)


def load_coco_from_buffer(json_buffer: str) -> COCO:
    with redirect_stdout(Verbose):
        coco = COCO()
        coco.dataset = json.loads(json_buffer)
        coco.createIndex()
    return coco


def load_coco_from_args(args):
    if args.input_coco:
        coco = COCO(args.input_coco)
    else:
        coco = load_coco_from_buffer(sys.stdin.read())
    return coco


def get_filter_list(csv_path):
    data_list = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_list.append(row[0])
    return data_list


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script filters a coco file based on a given text file list

    Usage: cat coco.json | coco-filter-by-list --filter-type image/id --txt-list image_ids.txt > coco.filtered.json
    """)
    parser.add_argument("--filter-type", type=str,
                        help="Options are 'image/id', 'image/path', 'category/id', 'category/name', 'category/supercategory', 'annotation/id'", required=True)
    parser.add_argument("--include-list", type=str, help="Path to the include text list")
    parser.add_argument("--exclude-list", type=str, help="Path to the exclude text list")
    parser.add_argument("--input-coco", type=str, help="Path to the input coco dataset. If not given, reads from stdin")
    parser.add_argument("--output-coco", type=str, help="Path to the output coco dataset. If not given, prints to stdout")
    args = parser.parse_args()
    return args


def include_list(args):
    coco = load_coco_from_args(args)
    outcoco = copy.deepcopy(coco.dataset)
    data_list = get_filter_list(args.include_list)

    if args.filter_type == "image/id":
        image_id_list = [int(iid) for iid in data_list]
        outcoco['images'] = coco.loadImgs(image_id_list)
        outcoco['annotations'] = coco.loadAnns(coco.getAnnIds(imgIds=image_id_list))
    elif args.filter_type == "annotation/id":
        ann_id_list = [int(aid) for aid in data_list]
        annotations = coco.loadAnns(ann_id_list)
        image_ids = [ann['image_id'] for ann in annotations]
        image_ids = list(set(image_ids))
        outcoco['images'] = coco.loadImgs(image_ids)
        outcoco['annotations'] = annotations
    elif args.filter_type == "image/path":
        image_id_list = []
        for image in coco.loadImgs(coco.getImgIds()):
            if image['path'] in data_list:
                image_id_list.append(image['id'])
        outcoco['images'] = coco.loadImgs(image_id_list)
        outcoco['annotations'] = coco.loadAnns(coco.getAnnIds(imgIds=image_id_list))
    else:
        raise NotImplementedError("TODO")

    if args.output_coco:
        json.dump(outcoco, open(args.output_coco, 'w'), indent=4)
    else:
        json.dump(outcoco, sys.stdout, indent=4)


def exclude_list(args):
    coco = load_coco_from_args(args)
    outcoco = copy.deepcopy(coco.dataset)
    data_list = get_filter_list(args.exclude_list)

    if args.filter_type == "image/id":
        exclude_ids = [int(iid) for iid in data_list]
        image_id_list = []
        for iid in coco.getImgIds():
            if iid not in exclude_ids:
                image_id_list.append(iid)
    elif args.filter_type == "image/path":
        image_id_list = []
        for image in coco.loadImgs(coco.getImgIds()):
            if image['path'] not in data_list:
                image_id_list.append(image['id'])

    outcoco['images'] = coco.loadImgs(image_id_list)
    outcoco['annotations'] = coco.loadAnns(coco.getAnnIds(imgIds=image_id_list))

    if args.output_coco:
        json.dump(outcoco, open(args.output_coco, 'w'), indent=4)
    else:
        json.dump(outcoco, sys.stdout, indent=4)


def main(args):
    if args.include_list and args.exclude_list:
        raise NotImplementedError("Including and Excluding not implemented yet")
    elif args.include_list:
        include_list(args)
    elif args.exclude_list:
        exclude_list(args)
    else:
        raise ValueError("--include-list or --exclude-list needs to be given")


if __name__ == "__main__":
    main(get_args())