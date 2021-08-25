#!/usr/bin/env python3


import argparse
import json
from pycocotools.coco import COCO
import os
def get_args():
    parser = argparse.ArgumentParser(description="Removes categories from a dataset")
    parser.add_argument('input_coco', help="The input coco file")
    parser.add_argument('output_coco', help="The output coco file")
    parser.add_argument('--remove-cat-ids', help="The category IDs to remove. Comma seperated")
    parser.add_argument('--remove-cat-names', help="The category names to remove. Comma separated")
    args = parser.parse_args()
    return args

def main(args):
    if not os.path.isfile(args.input_coco):
        raise ValueError("Coco %s not found " %args.input_coco)

    dataset = json.load(open(args.input_coco, 'r'))

    if args.remove_cat_ids and args.remove_cat_names:
        raise ValueError("Specify either remove cat ids or names")
    elif args.remove_cat_ids:
        cat_ids = [int(c) for c in args.remove_cat_ids.split(',')]
    elif args.remove_cat_names:
        cat_names = [c for c in args.remove_cat_names.split(',')]
        name2id_lookup = {}
        for cat in dataset['categories']:
            name2id_lookup[cat['name']] = cat['id']
        cat_ids = [name2id_lookup[c] for c in cat_names]
    else:
        raise ValueError("Specify ether remove cat ids or names")

    category_list = []
    annotation_list = []

    for cat in dataset['categories']:
        if cat['id'] not in cat_ids:
            category_list.append(cat)

    for ann in dataset['annotations']:
        if ann['category_id'] not in cat_ids:
            annotation_list.append(ann)

    dataset['categories'] = category_list
    dataset['annotations'] = annotation_list

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    image_list = []

    for image in dataset['images']:
        if len(coco.getAnnIds(imgIds=[image['id']])) > 0:
            image_list.append(image)


    dataset['images'] = image_list

    json.dump(dataset, open(args.output_coco, 'w'), sort_keys=True, indent=4)

if __name__ == "__main__":
    main(get_args())