#!/usr/bin/python3

import argparse
from pycocotools.coco import COCO
import numpy as np
import os
import glob
from PIL import Image
import logging
import json
import csv
import utm
import pymap3d
from tqdm import tqdm
import warnings
import habitat_modelling.utils.renavutils3 as rutil
from habitat_modelling.utils.raster_utils import extract_raster_patch, retrieve_pixel_coords, extract_geotransform

def get_args():
    parser = argparse.ArgumentParser(description="Creates a coco json from an ACFR Marine Dive. Puts dummy images in.")
    parser.add_argument('--renav-pose-file', '--poses', type=str, help="Path to the stereo_pose_est.data or vehicle_pose_est.data file which contains stereo poses")
    parser.add_argument('--trip', type=str, help="The trip this dive was on")
    parser.add_argument('--year', type=str, help="The year this dive was conducted")
    parser.add_argument('--cluster-file', type=str, help="The image label file generated from the clustering process. Usually called image_labels.data, in the renav directory")
    parser.add_argument('output_coco', help="Output path for the coco json")
    args = parser.parse_args()
    return args


class StereoPoseData:
    def __init__(self, pose_id, ts, lat, lon, x_north, y_east, z_depth, euler_x, euler_y, euler_z,
                 left_image_name, right_image_name, vehicle_altitude, bounding_image_radius=0.0, cross_over_point=False):
        self.pose_id = pose_id
        self.ts = ts
        self.lat = lat
        self.lon = lon
        self.x_north = x_north
        self.y_east = y_east
        self.z_depth = z_depth
        self.euler_x = euler_x
        self.euler_y = euler_y
        self.euler_z = euler_z
        self.left_image_name = left_image_name
        self.right_image_name = right_image_name
        self.vehicle_altitude = vehicle_altitude
        self.bounding_image_radius = bounding_image_radius
        self.cross_over_point = cross_over_point

class Datum:
    def __init__(self, lat, lon, alt=0.):
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt = float(alt)

def extract_stereo_poses(stereo_pose_file):
    """
    Extracts the stereo poses
    Args:
        stereo_pose_file: The stereo pose file, called stereo_pose_est.data in the renav directory

    Returns:
        list: pose_list, a list of all the stereo poses
        Datum: The datum

    """
    latitude = None
    longitude = None

    pose_list = []

    with open(stereo_pose_file, 'r') as data_file:
        # data_reader = csv.reader(data_file, delimiter='\t')
        data_reader = csv.reader(data_file, delimiter='\t')
        for row in data_reader:
            if row:
                if row[0][0] == '%':
                    continue
                elif row[0][0] == 'O':
                    if 'ORIGIN_LATITUDE' in row[0]:
                        latitude = row[0].split(' ')[-1]
                    elif 'ORIGIN_LONGITUDE' in row[0]:
                        longitude = row[0].split(' ')[-1]
                    continue
                # Try to handle spaces/tabs mixing
                if len(row) == 0:
                    row = [r for r in row.split(' ') if r != '']
                spd = StereoPoseData(
                    pose_id=int(row[0]),
                    ts=float(row[1]),
                    lat=float(row[2]),
                    lon=float(row[3]),
                    x_north=float(row[4]),
                    y_east=float(row[5]),
                    z_depth=float(row[6]),
                    euler_x=float(row[7]),
                    euler_y=float(row[8]),
                    euler_z=float(row[9]),
                    left_image_name=str(row[10]),
                    right_image_name=str(row[11]),
                    vehicle_altitude=float(row[12])
                )
                pose_list.append(spd)
    if latitude and longitude:
        datum = Datum(latitude, longitude)
    else:
        datum = None
    return pose_list, datum

def extract_cluster_labels(cluster_filepath):
    """
    Extracts the cluster labels from the cluster filepath.

    Args:
        cluster_filepath:

    Returns:

    """
    cluster_lookup = {} # key=image_fp, value=cluster label
    with open(cluster_filepath, 'r') as data_file:
        try:  # Try with a tab delimiter.
            data_reader = csv.reader(data_file, delimiter='\t')
            for row in data_reader:
                if row:
                    if row[0][0] == '%':
                        continue
                    # Remove the color/mono part and extension
                    fp = '_'.join(row[0].split('_')[:-1])
                    cluster_lookup[fp] = row[1]
        except IndexError:  # If out of range, try with a space delimiter
            data_reader = csv.reader(data_file, delimiter=' ')
            for row in data_reader:
                if row:
                    if row[0][0] == '%':
                        continue
                    # Remove the color/mono part and extension
                    # print(row)
                    fp = '_'.join(row[0].split('_')[:-1])
                    # print(fp)
                    cluster_lookup[fp] = row[1]
    return cluster_lookup


def main(args):
    # Parse the renav pose file. Gets the image poses and the datum
    stereo_pose_list, datum = extract_stereo_poses(args.renav_pose_file)

    cluster_lookup = extract_cluster_labels(args.cluster_file)

    image_list = []
    ann_list = []
    uniq_cats = []
    ann_lookup = []

    for n, spose in enumerate(stereo_pose_list):
        image = {
            "id": n,
            "file_name": spose.left_image_name,
            "path": spose.left_image_name,
            "width": 640,
            "height": 480,
            "pose": {
                "position": [spose.x_north, spose.y_east, spose.z_depth],
                "orientation": [spose.euler_x, spose.euler_y, spose.euler_z],
                "altitude": spose.vehicle_altitude
            },
            "geo_location": [spose.lat, spose.lon, -spose.z_depth]
        }

        reduced_fn = '_'.join(image['file_name'].split('_')[:-1])
        # print(list(cluster_lookup.keys())[0], reduced_fn)

        if reduced_fn not in cluster_lookup:
            warnings.warn("No annotation for image {}".format(image['file_name']))
            continue
        catID = cluster_lookup[reduced_fn]  # Label
        ann = {
            'id': n,
            'category_id': int(catID),
            'image_id': image['id'],
            "annotation_type": "point",  # Put it as a point label so it will show up in label software
            'bbox': [1, 2, 3, 4],  # Needs to be there
            "iscrowd": 0,
            "occluded": False,
            "segmentation": [[1, 2, 3, 4]],
            'area': 10.0
        }
        image_list.append(image)
        ann_list.append(ann)
        if catID not in uniq_cats:
            uniq_cats.append(catID)

    cat_list = []
    for catID in uniq_cats:
        cat = {
            "id": int(catID),
            "name": str(catID),
            "supercategory": ""
        }
        cat_list.append(cat)

    dataset = {
        "images": image_list,
        "annotations": ann_list,
        "categories": cat_list,

    }

    info = {}
    info['dive'] = args.renav_pose_file.split('/')[-3]
    if args.trip:
        info['trip'] = args.trip
    if args.year:
        info['year'] = args.year
    if datum:
        info['datum_latitude'] = datum.lat
        info['datum_longitude'] = datum.lon
        info['datum_altitude'] = datum.alt
    json.dump(dataset, open(args.output_coco, 'w'), indent=4, sort_keys=True)

if __name__ == "__main__":
    main(get_args())