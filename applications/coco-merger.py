#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from itertools import groupby

__author__ = 'Kent Hu, Jamie McColl, Toby Dunne'
__maintainer__ = 'Kent Hu'

DESCRIPTION = """
Read coco.json file(s) defined on command line
merge together 'images', 'annotations', 'categories' array.

Merges images and categories by id.
If there is a conflict:
 * if two images have the same id, and different paths, the second image will be renumbered and a warning produced.
    all annotations linked to the renumbered image will be updated to point to the new image_id
 * if two categories have the same id, and different names, the second category will be renumbered and a warning produced.
    all annotations linked to the renumbered category will be updated to point to the new category_id

Duplicate entries in the 'annotations' array will be removed 
    i.e. annotations with same image_id, category_id, bbox and segmentation)
Annotations id's in subsequent merged files will be numbered sequentially.

i.e. if dataset has two annotations ->
    first_ann = {                           |   second_ann = {
                  "id": 1,                  |                 "id": 2,
                  "category_id": 1,         |                 "category_id": 1,
                  "bbox": [0,0,0,0],        |                 "bbox": [0,0,0,0],
                  "segmentation" [ 0,0,0,0] |                 "segmentation" [ 0,0,0,0]
                }                           |                }

The second annotation will be removed as it has duplicate category id and bbox with the first annotation

Output merged coco json contents in stdout.

examples:
    coco-merge labelled/abc/coco.json labelled/def/coco.json

    use '-' to read from stdin    
    cat labelled/abc/coco.json | coco-merge - labelled/def/coco.json

    --append-categories:
        Consider two JSONs - a.json has two categories with ids 0 and 100, b.json has six categories with ids from 0 to 5.

        coco-merge a.json b.json --append-categories

            The resulting JSON will have category ids 0, 100, 101, 102, 103, 104, 105, with 0 and 100 matching the
            categories found in a.json.

        Switching the order:
        coco-merge b.json a.json --append-categories

            The resulting JSON will have category ids 0, 1, 2, 3, 4, 5, 6, 106, with 0 to 5 matching the categories
            found in b.json.
"""

def coco_is_valid(file_name, coco, mode):

    if mode == 'skip':
        return True

    full = mode == 'strict'

    is_valid = True

    try:
        if all([x not in coco for x in ['images', 'annotations', 'categories']]):
            logging.error("VALIDATION: coco must contain images annotations or categories")
            is_valid = False

        if 'annotations' in coco and not 'images' in coco:
            logging.error("VALIDATION: coco contains annotations with no images")
            is_valid = False

        if 'images' in coco:
            if any([not isinstance(x['id'], int) for x in coco['images']]):
                logging.error("VALIDATION: 1 or more images with id=None or not int")
                is_valid = False

            image_set = {el['id'] for el in coco['images']}
            if len(image_set) != len(coco['images']):
                logging.error("VALIDATION: duplicated image ids")
                is_valid = False

            # if any([not isinstance(x['width'], int) for x in coco['images']]):
            #     logging.error("VALIDATION: 1 or more images with invalid width")
            #     is_valid = False
            #
            # if any([not isinstance(x['height'], int) for x in coco['images']]):
            #     logging.error("VALIDATION: 1 or more images with invalid width")
            #     is_valid = False

        if 'annotations' in coco:
            if any([not isinstance(x['id'], int) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation with id=None or not int")
                is_valid = False

            if any(['image_id' not in x or not isinstance(x['image_id'], int) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation with image_id=None or not int")
                is_valid = False

            # annotation must have caption or category_id
            # if category_id is present, it must be an int
            if any([('caption' not in x and 'category_id' not in x)
                    or ('category_id' in x and not isinstance(x['category_id'], int)) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation without caption and invalid category_id")
                is_valid = False

            annotation_set = {el['id'] for el in coco['annotations']}
            if len(annotation_set) != len(coco['annotations']):
                logging.error("VALIDATION: duplicated annotation ids")
                is_valid = False

        if 'categories' in coco:
            if any([not isinstance(x['id'], int) for x in coco['categories']]):
                logging.error("VALIDATION: 1 or more categories with id=None or not int")
                is_valid = False

            category_set = {el['id'] for el in coco['categories']}
            if len(category_set) != len(coco['categories']):
                logging.error("VALIDATION: duplicated category ids")
                is_valid = False

        if full:
            # full checks cross reference of each annotation
            if 'annotations' in coco and len(coco['annotations']) > 0:
                for ann in coco['annotations']:
                    if ann['image_id'] not in image_set:
                        logging.error("VALIDATION: annotation id {} refers to missing image id {}".format(ann['id'], ann['image_id']))
                        is_valid = False

                for ann in coco['annotations']:
                    if 'category_id' not in ann and 'caption' not in ann:
                        logging.error("VALIDATION: annotation id:{} without category_id or caption".format(ann['id']))
                        is_valid = False
                    elif 'category_id' in ann and ann['category_id'] not in category_set:
                        logging.error("VALIDATION: annotation id:{} refers to missing category id {}".format(ann['id'], ann['category_id']))
                        is_valid = False
                        break

    except Exception as ex:
        logging.error("Exception while validating {}".format(ex))
        raise
        is_valid = False

    if not is_valid:
        logging.error("Error file is not valid coco: {}".format(file_name))

    return is_valid

def any_lambda(iterable, function):
    return any(function(i) for i in iterable)


def max_or_zero(i):
    return max(i) if len(i) > 0 else 0


def merge_coco(args, coco_list):
    coco_out = coco_list[0]
    for i in range(1, len(coco_list)):
        coco_out = merge_pair(args, coco_out, coco_list[i])
    return coco_out


def merge_pair(args, dest_coco, source_coco):
    dest_image_id_list = [e['id'] for e in dest_coco['images']]
    source_image_id_list = [e['id'] for e in source_coco['images']]

    dest_category_id_list = [e['id'] for e in dest_coco['categories']]
    source_category_id_list = [e['id'] for e in source_coco['categories']]

    max_image_id = max(max_or_zero(dest_image_id_list), max_or_zero(source_image_id_list))

    dest_max_category_id = max_or_zero(dest_category_id_list)
    max_category_id = max(dest_max_category_id, max_or_zero(source_category_id_list))

    # Merge categories
    dest_categories = dest_coco['categories']
    curr_cat_map = {el['id']: el for el in dest_categories}

    for category in source_coco['categories']:
        existing = curr_cat_map.get(category['id'], None)
        if args.append_categories:
            # append all categories, no merge
            old_category_id = category['id']
            category_id = dest_max_category_id + old_category_id + 1
            for ann in filter(lambda x: x.get('category_id', None) == old_category_id, source_coco['annotations']):
                ann['category_id'] = category_id
            category['id'] = category_id
            logging.info("Assigning new category id {} -> {}".format(old_category_id, category_id))
        elif existing is not None:
            if existing['name'] != category['name']:
                # duplicate category id - same id, different name
                max_category_id += 1
                old_category_id = category['id']
                for ann in filter(lambda x: x.get('category_id', None) == old_category_id, source_coco['annotations']):
                    ann['category_id'] = max_category_id
                category['id'] = max_category_id
                if args.merge_categories:
                    logging.info("Assigning new category id {} -> {}".format(old_category_id, max_category_id))
                else:
                    logging.error("Category conflict ID: {} '{}' != '{}'. "
                                  "Specify --append-categories or --merge-categories ".format(category['id'],
                                                                                              category['name'],
                                                                                              existing['name']))
                    exit(1)
            else:
                # same category_id, same name - skip second copy
                continue
        dest_categories.append(category)

    dest_coco['categories'] = dest_categories

    # merge images
    dest_images = dest_coco['images']
    dest_image_by_id = {el['id']: el for el in dest_images}

    for image in source_coco['images']:
        existing = dest_image_by_id.get(image['id'], None)
        if existing is not None:
            if existing['path'] != image['path']:
                # duplicate image id - same id, different path
                max_image_id += 1
                reassign_image_id = existing['id']
                for ann in filter(lambda x: x['image_id'] == reassign_image_id, source_coco['annotations']):
                    ann['image_id'] = max_image_id
                image['id'] = max_image_id
                logging.warning("Assigning new image id {} -> {}".format(reassign_image_id, max_image_id))
            else:
                # same image_id, same path - skip second copy
                continue
        dest_images.append(image)

    dest_coco['images'] = dest_images

    dest_annotations = dest_coco['annotations']

    dest_annotations_sorted = sorted(dest_annotations, key=lambda x: x['image_id'])
    dest_annotations_dict = dict([(key, group) for key, group in
                                  groupby(dest_annotations_sorted, lambda x: x['image_id'])])

    max_ann_id = max_or_zero([e['id'] for e in dest_coco['annotations']])
    source_annotations_sorted = sorted(source_coco['annotations'], key=lambda x: x['image_id'])
    for key, group in groupby(source_annotations_sorted, lambda x: x['image_id']):
        dest_group = dest_annotations_dict.get(key, None)
        if dest_group is None:
            for ann in group:
                max_ann_id += 1
                ann['id'] = max_ann_id
                dest_annotations.append(ann)
        else:
            for source_ann in group:
                if not any_lambda(dest_group, lambda x: x['image_id'] == source_ann['image_id']
                                                        and x.get('category_id', None) == source_ann.get('category_id',
                                                                                                         None)
                                                        and x.get('bbox', None) == source_ann.get('bbox', None)
                                                        and x.get('segmentation', None) == source_ann.get(
                    'segmentation', None)):
                    max_ann_id += 1
                    source_ann['id'] = max_ann_id
                    dest_annotations.append(source_ann)

    dest_coco['annotations'] = dest_annotations

    return dest_coco


def main(args):
    logging.basicConfig(
        format='%(filename)s: %(asctime)s.%(msecs)d: %(levelname)s: %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel,
    )
    coco_list = []

    if args.file_list:
        file_list = []
        for file_arg in args.files:
            if file_arg == '-':
                buffer = sys.stdin.read().strip()
            else:
                with open(file_arg, 'r') as f:
                    buffer = f.read().strip()
            file_list.extend(buffer.splitlines())
    else:
        file_list = args.files

    for file_arg in file_list:
        try:
            if file_arg == '-':
                curr_coco = json.load(sys.stdin)
            else:
                with open(file_arg, 'r') as f:
                    curr_coco = json.load(f)

            if not coco_is_valid(file_arg, curr_coco, args.validation):
                sys.exit(1)

            if curr_coco.get('images', None) is None:
                curr_coco['images'] = []
            if curr_coco.get('annotations', None) is None:
                curr_coco['annotations'] = []
            if curr_coco.get('categories', None) is None:
                curr_coco['categories'] = []

            coco_list.append(curr_coco)
        except json.decoder.JSONDecodeError as ex:
            logging.error("File is not valid JSON: {} {}".format(file_arg, ex))
            sys.exit(1)
        except Exception as ex:
            logging.error("Error loading: {} {}".format(file_arg, ex))
            sys.exit(1)

    if len(coco_list) == 0:
        logging.error("At least one input file is required")
        sys.exit(1)

    coco_out = merge_coco(args, coco_list)
    if args.output_coco:
        json.dump(coco_out, open(args.output_coco, 'w'), indent=args.indent)
    else:
        json.dump(coco_out, sys.stdout, indent=args.indent)
    return 0


def get_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'files',
        nargs='*',
        help="JSON files to merge together into one json (use - to read one file from stdin)",
    )
    parser.add_argument(
        '--file-list',
        action='store_true',
        help="specified files are are txt files with the JSON file names to merge (one file per line)",
    )
    parser.add_argument(
        '--append-categories',
        action='store_true',
        help='Assign category IDs in by appending, such that the ranges of IDs from different JSONs do not overlap\n'
             'e.g.\n'
             '[{ "id":0: "name": "A"}, { "id":1: "name": "B"}]\n'
             '[{ "id":0: "name": "A"}, { "id":1: "name": "B"}]\n'
             ' vvv\n'
             '[{ "id":0: "name": "A"}, { "id":1: "name": "B"}, { "id":2: "name": "A"}, { "id":3: "name": "B"},]'
    )
    parser.add_argument(
        '--merge-categories',
        action='store_true',
        help='Assign category IDs by merging (a warning is produced on re-assignment),\n'
             'e.g.\n'
             '[{ "id":0: "name": "A"}, { "id":1: "name": "B"}]\n'
             '[{ "id":0: "name": "X"}, { "id":1: "name": "B"}]\n'
             ' vvv\n'
             '[{ "id":0: "name": "A"}, { "id":1: "name": "B"}, { "id":2: "name": "X"}]'
    )
    parser.add_argument(
        '--minified',
        action='store_const',
        const=None,
        default=4,
        dest='indent',
        help="disable JSON pretty print",
    )
    parser.add_argument("--output-coco", help="Optionally write the output coco to a file")
    parser.add_argument('--validation',
                        type=str,
                        default='standard',
                        choices={'standard', 'strict', 'skip'},
                        help="Check if input coco is valid:\n"
                             "standard: quick checks(default)\n"
                             "strict: more in-depth cross-reference checks\n"
                             "skip: if you know coco is valid and want quicker execution.")

    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=logging.INFO,
        dest='loglevel',
        help="Verbose output to stderr",
    )
    logging_group.add_argument(
        '-d', '--debug',
        action='store_const',
        const=logging.DEBUG,
        dest='loglevel',
        help="Debug output to stderr",
    )
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(get_args()))
