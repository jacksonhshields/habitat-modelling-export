#!/usr/bin/env python3


import numpy as np
import argparse
try:
    from osgeo import gdal
except ImportError:
    import gdal
from osgeo import osr
import ast


def get_args():
    parser = argparse.ArgumentParser(description="Sets everything above water to no data value")
    parser.add_argument("input_raster", type=str, help="The input raster")
    parser.add_argument("output_raster", type=str, help="The output raster ")
    parser.add_argument("--remap", type=str, help="The classes to remap in format '{from:to}', e.g. to turn class 1 into 3 '{1:3}'")
    args = parser.parse_args()
    return args


def main(args):
    if args.input_raster == args.output_raster:
        raise ValueError("Input Raster and Output Raster cannot be the same!")

    input_ds = gdal.Open(args.input_raster)

    no_data_value = input_ds.GetRasterBand(1).GetNoDataValue()
    arr = input_ds.GetRasterBand(1).ReadAsArray().astype(int)
    
    remap = ast.literal_eval(args.remap)
    assert type(remap) == dict
    for c_from, c_to in remap.items():
        arr[arr == c_from] = c_to
    

    output_ds = gdal.GetDriverByName('GTiff').Create(args.output_raster, input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Float32)
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    output_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    output_ds.FlushCache()

if __name__ == "__main__":
    main(get_args())