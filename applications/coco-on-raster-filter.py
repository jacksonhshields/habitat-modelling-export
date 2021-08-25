#!/usr/bin/env python3
import os
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
import utm
from habitat_modelling.utils.raster_utils import extract_raster_patch, retrieve_pixel_coords, extract_geotransform


def get_args():
    parser = argparse.ArgumentParser(description="Creates a geographical output from a coco dataset")
    parser.add_argument('input_coco', help="The dataset that is to be filtered. A new copy of this dataset will have images removed around these areas.")
    parser.add_argument('output_coco', help="The dataset that is to be filtered. A new copy of this dataset will have images removed around these areas.")
    parser.add_argument('--rasters', help="The raster names used to do the checking", required=True)
    parser.add_argument('--raster-paths', help="The raster paths if the coco dataset doesn't have them")
    parser.add_argument('--raster-sizes', help="The raster names used to do the checking", required=True)
    args = parser.parse_args()
    return args





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
    coco = COCO(args.input_coco)

    raster_sizes = []
    for r in args.raster_sizes.split(','):
        raster_sizes.append([int(x) for x in r.split('x')])

    raster_collection = {}

    for n, rs in enumerate(args.rasters.split(',')):
        if args.raster_paths is None:
            raster_path = coco.dataset[rs]['path']
        else:
            raster_path = args.raster_paths.split(',')[n]

        raster_ds, geotransform, raster_info = extract_geotransform(raster_path)
        nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()

        raster_collection[rs] = {
            "dataset": raster_ds,
            "info": raster_info,
            "geotransform": geotransform,
            "size": raster_sizes[n],
            "no_data_val": nodataval
        }

    new_img_ids = []

    for image in coco.loadImgs(coco.getImgIds()):
        geoloc = image_geoloc(image)
        if geoloc is None:
            continue
        lla = geoloc

        rasters_ok = True

        for rs in raster_collection.keys():
            if 'UTM' in raster_collection[rs]['info']['projection']:  # TODO find less hacky way to determine
                use_utm = True
            else:
                use_utm = False

            if use_utm:
                ux, uy, zone_num, zone_letter = utm.from_latlon(lla[0], lla[1])
                px, py = retrieve_pixel_coords([ux, uy], list(raster_collection[rs]['geotransform']))
            else:
                px, py = retrieve_pixel_coords([lla[1], lla[0]], list(raster_collection[rs]['geotransform']))

            patch_size = raster_collection[rs]['size']

            half_patch = np.floor(patch_size[0])

            off_x = int(np.round(px - half_patch))
            off_y = int(np.round(py - half_patch))

            patch = extract_raster_patch(raster_collection[rs]['dataset'], off_x, off_y, patch_size[0])

            if patch is None or np.any(patch == raster_collection[rs]["no_data_val"]):
                rasters_ok = False
        if rasters_ok:
            new_img_ids.append(image['id'])

    output_coco = copy.deepcopy(coco.dataset)
    output_coco["images"] = coco.loadImgs(ids=new_img_ids)
    output_coco["annotations"] = coco.loadAnns(coco.getAnnIds(imgIds=new_img_ids))
    # os.makedirs(os.path.dirname(args.output_coco), exist_ok=True)
    json.dump(output_coco, open(args.output_coco, 'w'), sort_keys=True, indent=4)


if __name__ == "__main__":
    main(get_args())
