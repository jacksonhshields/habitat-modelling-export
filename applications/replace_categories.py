#!/usr/bin/env/python3

import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description="Removes categories from a dataset")
    parser.add_argument('input_coco', help="The input coco file")
    parser.add_argument('output_coco', help="The output coco file")
    parser.add_argument('--source-coco', help="The json to copy the categories from")
    args = parser.parse_args()
    return args

def main(args):
    if not os.path.isfile(args.input_coco):
        raise ValueError("Coco %s not found " %args.input_coco)

    dataset = json.load(open(args.input_coco, 'r'))

    srcds = json.load(open(args.source_coco,'r'))
    cats = srcds['categories']

    dataset['categories'] = cats

    json.dump(dataset, open(args.output_coco, 'w'), sort_keys=True, indent=4)

if __name__ == "__main__":
    main(get_args())