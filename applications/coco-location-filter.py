#!/usr/bin/env python3

import argparse
import simplekml
import numpy as np
import geojson
from pycocotools.coco import COCO
import webcolors
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
import pymap3d
import pandas as pd
import shapely
import shapely.geometry
import copy
import json

def get_args():
    parser = argparse.ArgumentParser(description="Creates a geographical output from a coco dataset")
    parser.add_argument('target_coco', help="The dataset that is to be filtered. A new copy of this dataset will have images removed around these areas.")
    parser.add_argument('filter_coco', help="The dataset that will do the filtering.")
    parser.add_argument('output_coco', help="Path to the output coco file")
    parser.add_argument('--exclusion-dist', type=float, default=20., help="The distance from the intersection points at which tp exclude data points from target coco")
    args = parser.parse_args()
    return args


def get_dataset_datum(dataset):
    if 'info' in dataset:
        if 'datum_latitude' in dataset['info'] and 'datum_longitude' in dataset['info']:
            datum = {}
            datum['latitude'] = dataset['info']['datum_latitude']
            datum['longitude'] = dataset['info']['datum_longitude']
            if 'datum_altitude' in dataset['info']:
                datum['latitude'] = dataset['info']['datum_altitude']
            else:
                datum['latitude'] = 0.0
        else:
            datum = None
    else:
        datum = None
    return datum



def get_pose_list(coco):
    if 'info' in coco.dataset:
        if 'datum_latitude' in coco.dataset['info'] and 'datum_longitude' in coco.dataset['info']:
            datum = {}
            datum['latitude'] = coco.dataset['info']['datum_latitude']
            datum['longitude'] = coco.dataset['info']['datum_longitude']
            if 'datum_altitude' in coco.dataset['info']:
                datum['latitude'] = coco.dataset['info']['datum_altitude']
            else:
                datum['latitude'] = 0.0
        else:
            datum = None
    pose_list = []
    for image in coco.loadImgs(coco.getImgIds()):
        if 'geo_location' in image:
            geoloc = image['geo_location']
        elif 'pose' in image:
            loc = image['pose']['position']
            geoloc = pymap3d.ned2geodetic(image['pose']['position'][0],
                                          image['pose']['position'][1],
                                          image['pose']['position'][2],
                                          datum['latitude'],
                                          datum['longitude'],
                                          datum['latitude'], ell=None, deg=True)
        else:
            continue
        pose_list.append((geoloc[0], geoloc[1]))  # Make tuple for shapely
    return pose_list


def image_geoloc(image, datum=None):
    if 'geo_location' in image:
        geoloc = image['geo_location']
    elif 'pose' in image and datum is not None:
        loc = image['pose']['position']
        geoloc = pymap3d.ned2geodetic(image['pose']['position'][0],
                                      image['pose']['position'][1],
                                      image['pose']['position'][2],
                                      datum['latitude'],
                                      datum['longitude'],
                                      datum['latitude'], ell=None, deg=True)
    else:
        geoloc = None
    return geoloc

def main(args):
    target_coco = COCO(args.target_coco)
    filter_coco = COCO(args.filter_coco)

    target_poses = get_pose_list(target_coco)
    filter_poses = get_pose_list(filter_coco)

    # Create shapely line strings
    target_ls = shapely.geometry.LineString(target_poses)
    filter_ls = shapely.geometry.LineString(filter_poses)

    # Get intersection points
    intersections = target_ls.intersection(filter_ls)

    intersections = np.array(intersections)

    intersections = intersections.reshape(-1, 2)  # Ensure data is Nx2

    datum = get_dataset_datum(target_coco.dataset)
    if datum is None:
        raise NotImplementedError("TODO. Pick a new datum")

    intersections_local = np.zeros(intersections.shape)
    for n in range(intersections.shape[0]):
        ned = pymap3d.geodetic2ned(intersections[n,0], intersections[n,1], 0.0, datum['latitude'], datum['longitude'], datum['latitude'], ell=None, deg=True)
        intersections_local[n,:] = ned[:2]

    id_list = []

    for image in target_coco.loadImgs(target_coco.getImgIds()):
        geoloc = image_geoloc(image, datum=datum)
        geoloc = np.array(geoloc[:2])

        ned = pymap3d.geodetic2ned(geoloc[0], geoloc[1], 0.0, datum['latitude'], datum['longitude'],
                                   datum['latitude'], ell=None, deg=True)
        loc = ned[:2]

        excluded = False

        for n in range(intersections.shape[0]):
            dist = np.sqrt(np.sum((loc - intersections_local[n])**2))
            excluded = False
            if dist < args.exclusion_dist:
                excluded = True
                break

        if not excluded:
            id_list.append(image['id'])

    if len(id_list) == 0:
        raise ValueError("Output dataset is empty")
    # Create output dataset
    output_ds = copy.deepcopy(target_coco.dataset)
    output_ds['images'] = target_coco.loadImgs(ids=id_list)
    output_ds['annotations'] = target_coco.loadAnns(target_coco.getAnnIds(imgIds=id_list))
    # keep categories the same


    json.dump(output_ds, open(args.output_coco, 'w'), indent=4, sort_keys=True)

if __name__ == "__main__":
    main(get_args())
