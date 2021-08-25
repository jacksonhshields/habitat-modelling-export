#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import argparse
import json
import glob
import warnings

def get_args():
    parser = argparse.ArgumentParser(description="Creates a coco json from an ACFR Marine Dive")
    parser.add_argument('category_path', help="Path to categories json with count item. Can be globbed")
    parser.add_argument('--num-classes', type=int, help="The total number of classes")
    parser.add_argument('output_csv', help="Path to the output csv")
    args = parser.parse_args()
    return args


def main(args):
    if os.path.isdir(args.category_path):
        args.category_path = args.category_path + '/*'

    files = glob.glob(args.category_path, recursive=True)


    data = {}
    for f in files:
        print("Extracting %s" %f)
        cats = json.load(open(f, 'r'))
        counts = np.zeros([args.num_classes])
        for cat in cats:
            if cat['id'] < args.num_classes and cat['id'] >= 0:
                counts[cat['id']] = cat['count']
            else:
                warnings.warn('category id %d is out of range' %cat['id'])
        data[os.path.splitext(os.path.basename(f))[0]] = counts

    columns = ["cluster_" + str(x) for x in range(args.num_classes)]
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    df.to_csv(args.output_csv)


if __name__ == "__main__":
    main(get_args())

