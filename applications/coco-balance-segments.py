#!/usr/bin/env python3

import numpy as np
import json
import argparse
import os
import glob
import scipy.stats
import warnings
import random
import copy
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Roughly balances a bunch of segments")
    parser.add_argument('--cocos', nargs='+', help="Directory containing all the coco files", required=True)
    parser.add_argument('--partitions', help="The names of each partition, comma separated, e.g. train,val,test", required=True)
    parser.add_argument('--ratios', help="The ratios of each partition, comma sperated, e.g. 0.5,0.25,0.25")
    parser.add_argument('--no-merge', action="store_false", help="Don't perform a merge")
    parser.add_argument('--copy-splits', action="store_true", help="Copy the splits to the output directory.")
    parser.add_argument("output_dir", help="The output directory")
    args = parser.parse_args()
    return args

def get_counts_array(categories, num_cats):
    counts = np.zeros(num_cats)
    for cat in categories:
        counts[cat['id']] = cat['count']
    return counts

def merge_part(cocos, output_coco):

    # TODO - don't use merge script...
    merge_script = "~/src/jhs/habitat-modelling/applications/coco-merger.py"
    merge_script = os.path.expanduser(merge_script)
    if not os.path.isfile(merge_script):
        merge_script = "~/src/jhs/habitat-modelling/applications/coco-merger.py"
        merge_script = os.path.expanduser(merge_script)
    if not os.path.isfile(merge_script):
        raise ValueError("Merge script not found")

    merge_cmd = ["python3", merge_script, *cocos, "--output-coco",
                 output_coco]
    # merged_ds = open(os.path.join(dataset_dir, "train.json"), 'w')
    # c = subprocess.Popen(merge_cmd, stdout=merged_ds)
    # c.wait()
    subprocess.check_output(merge_cmd)

def copy_file(from_fp, to_fp):
    """
    Performs an os copy.

    Args:
        from_fp: (str) the filepath to copy from
        to_fp: (str) the filepath to copy to

    Returns:
        None
    """
    subprocess.check_output(["cp", from_fp, to_fp])

def select_partition(pregistry, num_datasets):
    """
    Selects a partition that most needs more data to balance it
    Args:
        pregistry: (dict) The partition registry.
        num_datasets: (int) The total number of datasets

    Returns:
        str: The name of the selected partition. Used to key the partition registry dictionary.
    """
    # First select based on zero counts - if a partition has zero counts it should be first
    parts = []
    count_zeros = []
    for p,v in pregistry.items():
        count_zeros.append(len(v['counts']) - np.count_nonzero(v['counts']))
        parts.append(p)
    # Select idxs with most zeros. If there are all zeros or no zeros this will still work
    sel_idxs = np.argwhere(count_zeros == np.max(count_zeros)).flatten().tolist()

    if sel_idxs == 0:
        raise ValueError("No selected indexes")
    elif len(sel_idxs) == 1:
        return parts[sel_idxs[0]]

    # TODO - what about number of selections left??? - low selections should be preferenced
    # Select based on number of selections left
    use_leftover_selections = True
    if use_leftover_selections:
        parts2 = []
        leftover_selects = []
        for idx in sel_idxs:
            num_allowed = pregistry[parts[idx]]['ratio'] * num_datasets
            num_selected = len(pregistry[parts[idx]]['cocos'])
            parts2.append(parts[idx])
            leftover_selects.append(num_allowed - num_selected)
        sel_idxs = np.argwhere(leftover_selects == np.max(leftover_selects)).flatten().tolist()
    else:
        parts2 = parts

    if len(sel_idxs) == 0:
        raise ValueError("No selected indexes")
    elif len(sel_idxs) == 1:
        return parts2[sel_idxs[0]]
    
    # Option 1 - entropy - pick the least entropy count...
    parts3 = []
    ents = []
    for idx in sel_idxs:
        parts3.append(parts2[idx])
        count = pregistry[parts2[idx]]["counts"]
        if np.sum(count) == 0:
            ents.append(0.)
        else:
            count_scaled = np.array(count)/np.sum(count)
            ents.append(scipy.stats.entropy(count_scaled))
    
    return parts3[np.argmin(ents)]

def select_dataset(pentry, coco_registry, already_selected):
    """
    Selects a dataset that will most balance the pentry dataset.

    Args:
        pentry: (dict) The entry of the partition registry according to the selected partition
        coco_registry: (dict) The registry for all the coco datasets.
    Returns:
        int: the index for the selected dataset in the coco registry.
    """
    # First - select a dataset that will most reduce the number of zeros
    ccis = []
    count_zeros = []
    for ci, cv in coco_registry.items():
        if ci in already_selected:  # Check if the coco has already been selected by another process.
            continue
        if cv['path'] in pentry['cocos']:
            continue
        future_counts = pentry['counts'] + cv['counts']
        count_zeros.append(len(future_counts) - np.count_nonzero(future_counts))
        ccis.append(ci)
    # Select the indexes that correspond to the least number of zeros.
    sel_idxs = np.argwhere(count_zeros == np.min(count_zeros)).flatten().tolist()

    if len(sel_idxs) == 0:
        warnings.warn("No cocos left for selection")
        return None
    if len(sel_idxs) == 1:
        return ccis[sel_idxs[0]]

    # For the remaining datasets, pick one that will most balance the dataset.

    # entropy method...
    ccitmp = []
    for idx in sel_idxs:
        ccitmp.append(ccis[idx])
    ccis = ccitmp
    ents = []
    ccis2 = []
    for ci in ccis:
        count = coco_registry[ci]["counts"]
        if np.sum(count) == 0.:
            ents.append(0.)
        else:
            count_scaled = count / np.sum(count)
            ents.append(scipy.stats.entropy(count_scaled))

        ccis2.append(ci)
    sel_idxs = np.argwhere(ents == np.min(ents)).flatten().tolist()
    if len(sel_idxs) == 0:
        print(sel_idxs)
        raise ValueError("No selected idxs")
    elif len(sel_idxs) == 1:
        return ccis[sel_idxs[0]]
    else:
        return ccis[random.choice(sel_idxs)]

def main(args):
    if len(args.cocos) == 1:
        if os.path.isdir(args.cocos[0]):
            coco_paths = glob.glob(args.cocos[0] + '/*')
        else:
            coco_paths = args.cocos
    else:
        coco_paths = args.cocos
    
    # Find the number of categories
    num_cats = 0
    for cp in coco_paths:
        cc = json.load(open(cp, 'r'))
        if len(cc['categories']) > num_cats:
            num_cats = len(cc['categories'])
    
    coco_registry = {}
    for idx, cp in enumerate(coco_paths):
        cc = json.load(open(cp, 'r'))
        coco_registry[idx] = {
            "path": cp,
            "counts": get_counts_array(cc['categories'], num_cats=num_cats)
        }
    
    partitions = args.partitions.split(',')
    ratios = [float(x) for x in args.ratios.split(',')]
    part_registry = {}
    for part, ratio in zip(partitions, ratios):
        part_registry[part] = {
            "cocos": [],
            "counts": np.zeros(num_cats),
            "ratio": ratio
        }
    

    selected_idxs = []
    while len(selected_idxs) < len(coco_registry.keys()):
        # Select the most in need partition.
        part = select_partition(part_registry, len(coco_registry.keys()))

        # Select the dataset that most helps the partition.
        cidx = select_dataset(part_registry[part], coco_registry, selected_idxs)

        # Add the selected coco to the partition
        part_registry[part]["cocos"].append(coco_registry[cidx]["path"])

        part_registry[part]["counts"] += coco_registry[cidx]["counts"]

        # Append the selected indexes
        selected_idxs.append(cidx)

    os.makedirs(args.output_dir, exist_ok=True)

    printable_pregistry = {}
    for part, pv in part_registry.items():
        pv2 = copy.deepcopy(pv)
        pv2["counts"] = [int(x) for x in pv["counts"]]
        printable_pregistry[part] = pv2
    json.dump(printable_pregistry, open(os.path.join(args.output_dir, 'balance-segments-summary.json'), 'w'), indent=4)


    if args.no_merge is not False:
        for part, pv in part_registry.items():
            merge_part(pv['cocos'], os.path.join(args.output_dir, part + ".json"))

    if args.copy_splits:
        for part, pv in part_registry.items():
            part_dir = os.path.join(args.output_dir, part)
            os.makedirs(part_dir, exist_ok=True)
            for coco_path in pv['cocos']:
                copy_file(coco_path, os.path.join(part_dir, os.path.basename(coco_path)))











if __name__ == "__main__":
    main(get_args())