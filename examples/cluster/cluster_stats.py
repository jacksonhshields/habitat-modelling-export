#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

def get_args():
    parser = argparse.ArgumentParser(description="Displays statistics from clustering")
    parser.add_argument('--pickle', help="Path to saved_data.pkl from raster_cluster_geoinfer.py", required=True)
    parser.add_argument('--colormap', default='spectral', help="The colorbar to use")
    parser.add_argument('--save-dir', help="Directory to save confusion matrices")
    parser.add_argument('--show', action="store_true", help="Whether to show the plots")
    args = parser.parse_args()
    return args

def get_colormap(colormap):
    if colormap == 'spectral':
        clist = [(0, "red"), (0.25, "orange"), (0.5, "yellow"),
                 (0.75, "green"), (1, "blue")]
    else:
        raise NotImplementedError("Only 'spectral' colorbar implemented")
    rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
    return rvb

def main(args):
    data = pickle.load(open(args.pickle, 'rb'))
    cd_df = pd.DataFrame.from_dict({"cluster": data['cluster_labels'], "depth": data['depth'].squeeze()})

    cluster_idxs = [c for c in range(cd_df['cluster'].max()+1)]
    cluster_counts = [np.sum(np.argwhere(cd_df['cluster'] == c)) for c in cluster_idxs]
    cluster_counts_norm = cluster_counts / np.sum(cluster_counts)

    rvb = get_colormap(args.colormap)


    plt.figure()
    plt.bar(cluster_idxs, cluster_counts, color=rvb(np.array(cluster_idxs).astype(float)/len(cluster_idxs)))
    plt.xticks(cluster_idxs, tuple([str(c) for c in cluster_idxs]))
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title("Cluster Counts")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.savefig(os.path.join(args.save_dir, 'cluster_counts.png'))


    plt.figure()
    depth_pos = True
    cluster_depths = [cd_df[cd_df.cluster == c]['depth'].mean() for c in cluster_idxs]
    cluster_depths_var = [cd_df[cd_df.cluster == c]['depth'].std() for c in cluster_idxs]
    if depth_pos:
        cluster_depths = [abs(d) for d in cluster_depths]
    plt.bar(cluster_idxs, cluster_depths, color=rvb(np.array(cluster_idxs).astype(float)/len(cluster_idxs)), yerr=cluster_depths_var)
    plt.xticks(cluster_idxs, tuple([str(c) for c in cluster_idxs]))
    plt.xlabel('Cluster')
    plt.ylabel('Mean Depth')
    plt.title("Cluster Depths")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.savefig(os.path.join(args.save_dir, 'cluster_depths.png'))

    if args.show:
        plt.show()

if __name__ == "__main__":
    main(get_args())
