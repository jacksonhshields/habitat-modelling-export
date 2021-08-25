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
import kdtree

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
            datum = [dataset['info']['datum_latitude'], dataset['info']['datum_longitude']]
        else:
            datum = None
    else:
        datum = None
    return datum

class TreeNode(object):
    """
    Used within the KD Tree for quick lookup.
    """
    def __init__(self, x, y, id):
        self.coords = (x, y)
        self.id = id

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return 'TreeNode ({}, {}, {})'.format(self.coords[0], self.coords[1], self.id)

def build_tree(filter_coco, datum=None):
    nodes = []
    for image in filter_coco.loadImgs(filter_coco.getImgIds()):
        if datum is None:
            datum = image['geo_location']
        geoloc = image['geo_location']
        e,n,u = pymap3d.geodetic2enu(lat=geoloc[0], lon=geoloc[1], h=0., lat0=datum[0], lon0=datum[1], h0=0.)
        tn = TreeNode(e,n,image['id'])
        nodes.append(tn)
    tree = kdtree.create(nodes)
    return tree, datum



def main(args):
    target_coco = COCO(args.target_coco)
    filter_coco = COCO(args.filter_coco)


    tree, datum = build_tree(filter_coco, get_dataset_datum(filter_coco.dataset))

    id_list = []

    for image in target_coco.loadImgs(target_coco.getImgIds()):
        geoloc = image['geo_location']
        geoloc = np.array(geoloc[:2])
        enu = pymap3d.geodetic2enu(geoloc[0], geoloc[1], 0., datum[0], datum[1], 0.)
        nearest = tree.search_nn_dist([enu[0], enu[1]], distance=args.exclusion_dist)
        if len(nearest) == 0:
            id_list.append(image['id'])
        else:
            print("rejected %d" % image['id'])

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
