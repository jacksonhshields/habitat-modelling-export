#!usr/bin/env python3

import argparse
import numpy as np
from pycocotools.coco import COCO
import copy
import json
import osr
from habitat_modelling.utils.raster_utils import extract_raster_patch, extract_geotransform, retrieve_pixel_coords, retrieve_geo_coords
from habitat_modelling.utils.raster_utils import retrieve_pixel_coords, extract_raster_patch, extract_geotransform, get_ranges

import utm

def get_args():
    parser = argparse.ArgumentParser(description="Ensures all coco paths have image/w")
    parser.add_argument('--rasters', help="The name of each raster. Comma separated e.g. bathymetry,backscatter")
    parser.add_argument('--raster-paths', help="The path of each raster. Comma separated, e.g. /path/to/bathymetry.tif,/path/to/backscatter.tif")
    parser.add_argument('--sample-rate', default=1.0, type=float, help="The sample rate to use when extracting points via a grid")
    parser.add_argument('--x-range', type=str,
                        help="The range of x values to extract from the dataset. Must match the units of the raster, E.g. 6200.5,12500.3 to extract everything between 6200.5 to 12500.3")
    parser.add_argument('--y-range', type=str, help="The range of y values to extract from the dataset. Must match the units of the data")
    parser.add_argument('--patch-sizes', type=str, default="1x1", help="The size of each patch to extract")
    parser.add_argument('--check-patches', action="store_true", help="Set this if you don't want to check the patch validity. Not recommended")
    parser.add_argument('output_coco', help="Path to the output coco file")
    args = parser.parse_args()
    return args

def main(args):

    # ------------------------
    # Check + Enter Args
    # ------------------------

    if args.sample_rate > 1.0 or args.sample_rate < 0.0:
        raise ValueError("Sample ratio should be between 0 and 1")

    if args.x_range and args.y_range:
        x_range = [float(x) for x in args.x_range.split(',')]
        y_range = [float(y) for y in args.y_range.split(',')]
    else:
         x_range = None
         y_range = None

    rasters = args.rasters.split(',')
    raster_paths = args.raster_paths.split(',')
    raster_collection = {}
    raster_sizes = []
    for r in args.patch_sizes.split(','):
        raster_sizes.append([int(x) for x in r.split('x')])

    # ----------------------
    # Get Raster Datasets
    # ----------------------
    for n, rs in enumerate(rasters):
        # Check the raster is in the dataset

        raster_ds, geotransform, raster_info = extract_geotransform(raster_paths[n])
        # Get no data value
        nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()
        raster_collection[rs] = {
            "dataset": raster_ds,
            "info": raster_info,
            "geotransform": geotransform,
            "size": raster_sizes[n],
            "no_data_val": nodataval
        }


    # ----------------------------------------------
    # Set leading raster = first raster
    # ----------------------------------------------
    leading_raster = raster_collection[rasters[0]]

    # ------------------------
    # Get range + samples
    # ------------------------
    x_samples, y_samples, num_x_samples, num_y_samples, x_range, y_range = get_ranges(x_range, y_range, args.sample_rate, leading_raster['geotransform'], leading_raster['info'])

    image_list = []
    ann_list = []
    image_count = 0

    for i in range(num_x_samples):
        for j in range(num_y_samples):
            x_base = x_samples[i]
            y_base = y_samples[j]

            if args.check_patches:
                x = x_base
                y = y_base

                off_x = int(x - leading_raster['size'][0] / 2)
                off_y = int(y - leading_raster['size'][1] / 2)


                patch = extract_raster_patch(leading_raster['dataset'], off_x, off_y, leading_raster['size'][0])

                #

                if patch is None or np.max(patch) > 1e8 or np.min(patch) < -1e8 or np.any(np.isclose(patch, leading_raster["no_data_val"])):  # TODO no data values
                    continue

            gx, gy = retrieve_geo_coords((x_base,y_base), list(leading_raster['geotransform']))
            if 'UTM' in leading_raster['info']['projection']:
                srs = osr.SpatialReference(wkt=leading_raster['info']['projection'])
                projcs = srs.GetAttrValue('projcs')
                if 'zone' in projcs:
                    zonestr = projcs.split(' ')[-1]
                    zone_num = int(zonestr[:2])
                    zone_hem = zonestr[-1]
                else:
                    raise ValueError("Projection doesn't contain zone")
                latlon = list(utm.to_latlon(gx, gy, zone_num, zone_hem))

            else:
                latlon = [gx, gy]  # TODO check this output
            lla = latlon + [0.]

            image = {
                "path": "dummy",
                "id": image_count,
                "geo_location": lla
            }
            image_list.append(image)
            ann = {
                "type": "point",
                "id": image_count,
                "category_id": 0,
                "image_id": image_count
            }
            ann_list.append(ann)
            image_count += 1


    coco_dataset = {
        "images": image_list,
        "annotations": ann_list,
        "categories": [],
    }
    for raster, path in zip(rasters, raster_paths):
        coco_dataset[raster] = {
            "path": path
        }
    json.dump(coco_dataset, open(args.output_coco, 'w'), indent=4, sort_keys=True)


if __name__ == "__main__":
    main(get_args())