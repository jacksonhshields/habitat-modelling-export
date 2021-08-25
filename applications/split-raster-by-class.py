#!/usr/bin/env python3
import argparse
import os

import numpy as np
try:
    from osgeo import gdal
except ImportError:
    import gdal
from habitat_modelling.utils.raster_utils import extract_geotransform


def get_args():
    parser = argparse.ArgumentParser(description="Splits a raster by class")
    parser.add_argument("--raster", required=True, type=str, help="The input raster to split")
    parser.add_argument("--output-dir", required=True, type=str, help="The output directory for the raster")
    args = parser.parse_args()
    return args


def main(args):
    raster_ds, geotransform, raster_info = extract_geotransform(args.raster)
    category_array = raster_ds.GetRasterBand(1).ReadAsArray()
    category_array = category_array.astype(np.int16)
    os.makedirs(args.output_dir, exist_ok=True)
    for c in range(category_array.max()):
        split_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(args.output_dir, '%d.tif' %c),
                                                          raster_info['size'][0],
                                                          raster_info['size'][1],
                                                          1, gdal.GDT_Float32)
        class_array = category_array == c
        split_ds.SetGeoTransform(geotransform)
        split_ds.SetProjection(raster_ds.GetProjection())
        split_ds.GetRasterBand(1).WriteArray(class_array.astype(np.float32))
        split_ds.GetRasterBand(1).SetNoDataValue(-1.0)
        split_ds.FlushCache()  # write to disk
        split_ds = None


if __name__ == "__main__":
    main(get_args())