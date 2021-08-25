#!/usr/bin/env python3

import argparse
import json
from habitat_modelling.utils.raster_utils import extract_geotransform
import numpy as np
from pycocotools.coco import COCO

def get_args():
    parser = argparse.ArgumentParser(description="Splits the dataset into geographic quadrants")
    parser.add_argument('--input-coco','-i',required=True,type=str,help="Input coco dataset")
    parser.add_argument('--x-divs',required=True,type=int,help="Number of divisions in the x direction")
    parser.add_argument('--y-divs',required=True,type=int,help="Number of divisions in the y direction")
    args = parser.parse_args()
    return args

def get_corners(coco_path):
    coco = COCO(coco_path)
    locs = []
    for image in coco.loadImgs(coco.getImgIds()):
        if 'geo_location' in image:
            locs.append(image['geo_location'][:2])
    loc_array = np.asarray(locs)
    max_lat = loc_array[:,0].max()
    min_lat = loc_array[:,0].min()
    min_lon = loc_array[:,1].min()
    max_lon = loc_array[:,1].max()
    return min_lat, min_lon, max_lat, max_lon



def main(args):

    min_lat, min_lon, max_lat, max_lon = get_corners(args.input_coco)

    x_points = np.linspace(min_lon, max_lon,args.x_divs+1)
    y_points = np.linspace(min_lat, max_lon,args.y_divs+1)

    q_count = 0
    data = {}
    for i in range(len(x_points)-1):
        for j in range(len(y_points)-1):
            l = x_points[i]
            r = x_points[i+1]
            u = y_points[j]
            d = y_points[j+1]
            corners = "[[%f,%f],[%f,%f],[%f,%f],[%f,%f]]" % (u,l,u,r,d,r,d,l)
            with open('corner_%d.txt' % q_count, 'w') as out_file:
                out_file.write(corners)
            q_count += 1


if __name__ == "__main__":
    main(get_args())
