#!/bin/bash

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from habitat_modelling.utils.display_utils import plot_confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description="Plots better confusion matrices")
    parser.add_argument('--pickle', help="Path to pickle containing the count dictionaries. From the validate_neighbours script", required=True)
    parser.add_argument('--classes', help="Comma separated classes", required=True)
    parser.add_argument('--norm', action="store_true", help="Normalise")
    parser.add_argument('--save-dir', help="Directory to save confusion matrices")
    args = parser.parse_args()
    return args

def main(args):
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    all_conf_dicts = pickle.load(open(args.pickle, 'rb'))
    for k,v in all_conf_dicts.items():
        if args.save_dir:
            save_path = os.path.join(args.save_dir, k + '.png')
        else:
            save_path = None
        df = pd.DataFrame.from_dict(v, orient='index',columns=args.classes.split(','))
        plot_confusion_matrix(df.to_numpy().astype(np.int32), df.columns.values,
                              normalized=args.norm,
                              save_path=save_path)


if __name__ == "__main__":
    main(get_args())