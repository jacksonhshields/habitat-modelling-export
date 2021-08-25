#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="Plots the maximum validation accuracy for the iterative training")
    parser.add_argument('--results', help="The name of each result", required=True)
    parser.add_argument('--results-dirs', help="Path to the results directory, containing iteration_0, iteration_1 etc.", required=True)
    parser.add_argument('--save-path', help="Optionally save the plot")
    args = parser.parse_args()
    return args


def main(args):
    plt.figure()
    for result_name, results_dir in zip(args.results.split(','), args.results_dirs.split(',')):
        all_dirs = glob.glob(results_dir + '/iteration*')
        all_results = []
        for path in all_dirs:
            log_path = os.path.join(results_dir, path, 'logs', 'validation_logs.csv')
            if not os.path.isfile(log_path):
                continue
            df = pd.read_csv(log_path)
            iteration_num = int(path.split('_')[-1].replace('/',''))
            results = np.array([iteration_num, df['Val_Acc'].max()])
            all_results.append(results)
        all_results = np.array(all_results)

        all_results = all_results[all_results[:,0].argsort(axis=0),:]
        plt.plot(all_results[:,0], all_results[:,1], label=result_name)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title("Performance of selection criteria")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(get_args())


