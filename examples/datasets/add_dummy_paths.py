#!/usr/bin/env python3


import argparse
import os
import json


def get_args():
    parser = argparse.ArgumentParser(description="Adds dummy image paths to perform nicely with coco merge")
    parser.add_argument('input_cocos', help="The input cocos, comma separated")
    parser.add_argument('output_dir', help="Path to the output coco")
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for n,ds_path in enumerate(args.input_cocos.split(',')):
        ds = json.load(open(ds_path,'r'))
        image_list = []
        for i, image in enumerate(ds['images']):
            image["path"] = "%d_%d" % (n, i)
            image_list.append(image)
        ds['images'] = image_list
        json.dump(ds, open(os.path.join(args.output_dir, os.path.basename(ds_path)), 'w'), indent=4)
if __name__ == "__main__":
    main(get_args())
