#!/usr/bin/env python3

import argparse
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Creates a geographical output from a coco dataset")
    parser.add_argument('results_json', help="The results json")
    parser.add_argument('--save-dir', help="The directory to save the images to")
    parser.add_argument('--plot', action="store_true", help="Where to save the images to")
    parser.add_argument('--classes', help='The class names, comma separated')
    args = parser.parse_args()
    return args

def main(args):
    results = json.load(open(args.results_json, 'r'))
    if args.classes is None:
        num_classes = len(results["bench_accuracy_per_class"])
        classes = [str(x) for x in range(num_classes)]
    else:
        classes = args.classes.split(',')
    bench_acc_pc = pd.DataFrame([results["bench_accuracy_per_class"]], columns=classes)
    label_acc_pc = pd.DataFrame([results["label_accuracy_per_class"]], columns=classes)
    neighbour_acc_pc = pd.DataFrame([results["neighbour_accuracy_per_class"]], columns=classes)

    bench_fig = bench_acc_pc.plot.bar()
    bench_fig.set_title('Benchmark Accuracy Per Class')
    bench_fig.axis('tight')
    label_fig = label_acc_pc.plot.bar()
    label_fig.set_title('Label Accuracy Per Class')
    neighbour_fig = neighbour_acc_pc.plot.bar()
    neighbour_fig.set_title('Neighbour Accuracy Per Class')

    # if args.save_dir:
    #     bench_fig.savefig(os.path.join(args.save_dir, "bench_accuracy_per_class.png"))
    #     label_fig.savefig(os.path.join(args.save_dir, "label_accuracy_per_class.png"))
    #     neighbour_fig.savefig(os.path.join(args.save_dir, "neighbour_accuracy_per_class.png"))


    if args.plot:
        plt.show(bench_fig)
        bench_fig.plot()
        label_fig.plot()
        neighbour_fig.plot()



if __name__ == "__main__":
    main(get_args())