#!/usr/bin/env python3

import argparse
import json
from habitat_modelling.utils.raster_utils import extract_geotransform, get_ranges, retrieve_geo_coords, extract_raster_patch
from osgeo import osr
import pymap3d
import utm
import warnings
import ast

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('label_raster', help="Path to the raster containing the labels")
    parser.add_argument('output_coco', help="The output coco file")
    parser.add_argument('--x-range', help="The range of x values to extract from the dataset. Must match the units of the data. Comma seperated")
    parser.add_argument('--y-range', help="The range of y values to extract from the dataset. Must match the units of the data. Comma seperated")
    parser.add_argument('--sample-rate', default=1.0, type=float,
                        help="The sample rate to use when extracting points via a grid")
    parser.add_argument('--cid-to-name', type=str, help="Dictionary converting cid to name, e.g. '{0:\"sand\", 1:\"coral\"}'")
    args = parser.parse_args()
    return args

def main(args):
    # Load the raster file
    label_raster, geotransform, raster_info = extract_geotransform(args.label_raster)
    no_data_val = label_raster.GetRasterBand(1).GetNoDataValue()
    # Test the sample rate
    if args.sample_rate > 1.0 or args.sample_rate < 0.0:
        raise ValueError("Sample ratio should be between 0 and 1")
    # Get the ranges out
    if args.x_range and args.y_range:
        x_range = [float(x) for x in args.x_range.split(',')]
        y_range = [float(y) for y in args.y_range.split(',')]
    else:
         x_range = None
         y_range = None


    # Get the samples
    x_samples, y_samples, num_x_samples, num_y_samples, x_range, y_range = get_ranges(x_range, y_range, args.sample_rate, geotransform, raster_info)
    # Iterate through all the sampled pixels
    images = []
    unique_cids = []
    annotations = []

    if args.cid_to_name is None:
        cid_to_name = None
    else:
        cid_to_name = ast.literal_eval(args.cid_to_name)

    image_count = 0
    for i in range(num_x_samples):
        for j in range(num_y_samples):
            x_base = x_samples[i]
            y_base = y_samples[j]
            gx, gy = retrieve_geo_coords((x_base,y_base), list(geotransform))
            if 'UTM' in raster_info['projection']:
                srs = osr.SpatialReference(wkt=raster_info['projection'])
                projcs = srs.GetAttrValue('projcs')
                if 'zone' in projcs:
                    zonestr = projcs.split(' ')[-1]
                    zone_num = int(zonestr[:2])
                    zone_hem = zonestr[-1]
                    if zone_hem == "N":
                        northern = True
                    elif zone_hem == "S":
                        northern = False
                    else:
                        raise ValueError("Zone hemisphere has to be either north or south")
                else:
                    raise ValueError("Projection doesn't contain zone")
                latlon = list(utm.to_latlon(gx, gy, zone_num, northern=northern))

            else:
                latlon = [gx, gy]  # TODO check this output
            lla = latlon + [0.]

            cat_id = extract_raster_patch(label_raster, x_base, y_base, 1)

            if cat_id is None or cat_id == no_data_val:
                continue
            cat_id = int(cat_id)

            if cid_to_name is None:
                if cat_id not in unique_cids:
                    unique_cids = cat_id
            else:
                if cat_id not in cid_to_name:
                    warnings.warn("cat id: %d not in cid_to_name" %cat_id)
                    continue
                else:
                    cname = cid_to_name[cat_id]

            image = {
                "path": "dummy",
                "id": image_count,
                "geo_location": lla,
                "width": 0,
                "height": 0,
                "filename": "dummy"
            }
            images.append(image)
            ann = {
                "type": "point",
                "id": image_count,
                "category_id": cat_id,
                "image_id": image_count
            }
            annotations.append(ann)
            image_count += 1

    # Create categories
    categories = []
    if cid_to_name is None:
        for cid in unique_cids:
            cat = {
                "id": cid,
                "name": str(cid),
                "supercategory": ""
            }
            categories.append(cat)
    else:
        for cid,cname in cid_to_name.items():
            cat = {
                "id": cid,
                "name": cname,
                "supercategory": ""
            }
            categories.append(cat)


    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    json.dump(dataset, open(args.output_coco, 'w'), indent=4, sort_keys=True)


















if __name__ == "__main__":
    main(get_args())