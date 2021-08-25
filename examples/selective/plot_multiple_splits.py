#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import glob
import os

def get_args():
    parser = argparse.ArgumentParser(description="Plots multiple results for training selections")
    parser.add_argument('experiment_dir', help="Path to the coco dataset to be used for validation")
    parser.add_argument('subdirs', help="Comma sepated subsections, e.g. part10_part20_part30,part15_part25,part35")
    parser.add_argument('--accuracy', type=str, default='label', help="The accuracy metric to use, options are {label,neighbour}")
    args = parser.parse_args()
    return args

def main(args):
    plt.figure()
    marker_symbols = ['x', 'o', 'd', '*']
    mode_titles = {
        "epistemic_mean": "Epistemic",
        "aleatoric_mean": "Aleatoric",
        "random": "Random"
    }
    mode_colours = {
        "epistemic_mean": 'g',
        "aleatoric_mean": 'b',
        "random": 'r'
    }

    for n, subdir in enumerate(args.subdirs.split(',')):
        marker_sym = marker_symbols[n]
        for mode in ["epistemic_mean", "aleatoric_mean", "random"]:
            results = []
            for iter_dir in glob.glob(os.path.join(args.experiment_dir, subdir, mode, "iteration*")):
                if not os.path.isfile(os.path.join(iter_dir, 'validation', 'results.json')):
                    raise ValueError("Validation needs to be run, results json not found %s" %os.path.join(iter_dir, 'validation', 'results.json'))
                results_dict = json.load(open(os.path.join(iter_dir, 'validation', 'results.json'),'r'))
                if args.accuracy == "label":
                    acc = results_dict["label_accuracy"]
                elif args.accuracy == "neighbour":
                    acc = results_dict["neighbour_accuracy"]
                else:
                    raise ValueError("Accuracy %s not supported" % args.accuracy)
                iter = int(os.path.basename(iter_dir).split('_')[-1])
                results.append(np.array([iter, acc]))
            all_results = np.asarray(results)
            all_results = all_results[all_results[:, 0].argsort(axis=0), :]
            marker = marker_sym + '-' + mode_colours[mode]
            plt.plot(all_results[:,0], all_results[:,1], marker, label=mode_titles[mode])

    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title("Performance of selection criteria")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()










if __name__ == "__main__":
    main(get_args())