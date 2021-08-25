#!/usr/bin/python3

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
    parser.add_argument('output_coco', help="Path to the output coco dataset")
    args = parser.parse_args()
    return args


def filename_to_timestamp(file_name):
    yyyymmdd = file_name.split('_')[1]
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])
    hhmmss = file_name.split('_')[2]
    hours = int(hhmmss[0:2])
    mins = int(hhmmss[2:4])
    secs = int(hhmmss[4:6])
    ms = int(file_name.split('_')[3])
    micro = ms*1000
    d = datetime.datetime(year, month, day, hours, mins, secs, micro)
    return d.timestamp()


def main(args):
    dataset = json.load(open(args.coco_path, 'r'))

    image_list = dataset['images']

    stamps = []

    for n,image in enumerate(image_list):
        time_stamp = filename_to_timestamp(image['file_name'])
        stamps.append(time_stamp)
    stamps = np.asarray(stamps)
    order = np.argsort(stamps)
    sorted_image_list = []

    for n in range(len(order)):
        idx = order[n]
        sorted_image_list.append(image_list[idx])

    dataset['images'] = sorted_image_list

    json.dump(dataset, open(args.output_coco, 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    main(get_args())
