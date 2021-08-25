#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Standardises numpy arrays and checks")
    parser.add_argument('latent_array', help="Path to the latent array")
    parser.add_argument('label_array', help="Path to the latent labels")
    parser.add_argument('--output-array', help="Path to the output latent array. If left blank overwrites original.")
    parser.add_argument('--output-labels', help="Path to the latent labels. If left blank overwrites original.")
    parser.add_argument('--scaler-output', help="Where to pickle the scaler to")
    args = parser.parse_args()
    return args

def main(args):
    x = np.load(args.latent_array)
    y = np.load(args.label_array)
    mask = np.any(np.isnan(x),axis=1)
    x = x[~mask]
    y = y[~mask]
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    if args.output_array:
        output_array_path = args.output_array
    else:
        output_array_path = args.latent_array
    if args.output_labels:
        output_label_path = args.output_labels
    else:
        output_label_path = args.label_array

    np.save(output_array_path, x)
    np.save(output_label_path, y)
    if args.scaler_output:
        pickle.dump(scaler, open(args.scaler_output, 'wb'))

if __name__ == "__main__":
    main(get_args())