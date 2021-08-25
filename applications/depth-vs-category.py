#!/usr/bin/env python3
import argparse
from pycocotools.coco import COCO
import json
import numpy as np
import warnings
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Creates a coco json from an ACFR Marine Dive")
    parser.add_argument('coco_path', help="Path to the coco format dataset. Needs to be ordered first!")
    parser.add_argument('--num-samples', type=int, help="Number of samples to plot")
    parser.add_argument('--sample-rate', type=float, help="Number of samples to plot")
    parser.add_argument('--x-axis', type=str, default="seq", help="The data to plot on the x-axis, options are {seq,time,dist}")
    parser.add_argument('--output-csv', type=str, help="The optional path to the output csv")
    parser.add_argument('--marker-size', type=float, default=5.0, help="Size of the markers")
    parser.add_argument('--save-plot', type=str, default="depth-vs-category.png", help="Depth vs Category")
    args = parser.parse_args()
    return args

def filename_to_timestamp(file_name):
    yyyymmdd = file_name.split('_')[1]
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])
    hhmmss = file_name.split('_')[2]
    hours = int(hhmmss[0:2])
    mins = int(hhmmss[2:4])
    secs = int(hhmmss[4:6])
    ms = int(file_name.split('_')[3])
    micro = ms*1000
    d = datetime.datetime(year, month, day, hours, mins, secs, micro)
    return d.timestamp()

def main(args):
    coco = COCO(args.coco_path)

    all_imgs_ids = coco.getImgIds()

    if args.num_samples:
        all_imgs_ids = np.asarray(all_imgs_ids)
        selected_idxs = np.linspace(0,len(all_imgs_ids)-1,args.num_samples).astype(np.int32)
        img_ids = list(np.asarray(all_imgs_ids[selected_idxs]))
    elif args.sample_rate:
        if args.sample_rate < 0.0 or args.sample_rate > 1.0:
            raise ValueError("Sample Rate should be between 0-1")
        all_imgs_ids = np.asarray(all_imgs_ids)
        selected_idxs = np.arange(0,len(all_imgs_ids), int(1/args.sample_rate))
        img_ids = list(np.asarray(all_imgs_ids[selected_idxs]))
    else:
        img_ids = all_imgs_ids

    # Initialise empty lists for the data
    dists = []
    prev_loc = None
    seqs = []
    times = []
    inital_time = None

    cats = []
    depths = []

    for n, image in enumerate(coco.loadImgs(img_ids)):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
        if len(anns) == 0:
            continue
        cat = anns[0]['category_id']
        if 'pose' in image:
            depth = -image['pose']['position'][2]
        elif 'geo_location' in image:
            depth = image['geo_location'][2]
        else:
            warnings.warn("No depth in image")
            continue
        cats.append(cat)
        seqs.append(n)
        depths.append(depth)

        # If the initial time has not been initialised,
        if inital_time is None:
            inital_time = filename_to_timestamp(image['file_name']) # initialise it
        times.append(filename_to_timestamp(image['file_name']) - inital_time)  # Append the timestamp to the list

        if 'pose' in image:  # Distance only possible if pose in the image
            # If the previous location has not been initialised, set it
            if prev_loc is None:
                prev_loc = np.array(image['pose']['position'][:2])  # Set the previous location as the starting location
            loc = np.array(image['pose']['position'][:2])  # Previous array position
            dist = np.sqrt(np.sum((loc-prev_loc)**2))  # Euclidian distance
            dists.append(dist)  # Append to the array
            prev_loc = loc  # Set the previous location as the current
    data = {
        "cat": np.asarray(cats),
        "seq": np.asarray(seqs),
        "depth": np.asarray(depths),
        "time": np.asarray(times),
        "dist": np.cumsum(np.asarray(dists))
    }

    df = pd.DataFrame.from_dict(data)

    if args.output_csv:
        df.to_csv(args.output_csv)

    if args.x_axis == "seq":
        x_data = data['seq']
        x_title = 'Sequence'
    elif args.x_axis == "time":
        x_data = data['time']
        x_title = 'Time (s)'
    elif args.x_axis == "dist":
        x_data = data['dist']
        x_title = "Distance (m)"
    else:
        raise ValueError("X Axis needs to be one of {seq,time,dist")
    plt.figure(figsize=(14, 6))
    plt.scatter(x_data, data['depth'], s=args.marker_size, c=data['cat'], cmap='jet')
    plt.colorbar()
    plt.ylabel("Depth (m)")
    plt.xlabel(x_title)
    plt.title("Depth vs Category")
    if args.save_plot:
        plt.savefig(args.save_plot)
    plt.show()




if __name__ == "__main__":
    main(get_args())