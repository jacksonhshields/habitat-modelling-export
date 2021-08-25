#!/usr/bin/env python3


import numpy as np
import argparse
try:
    from osgeo import gdal
except ImportError:
    import gdal
from osgeo import osr

def get_args():
    parser = argparse.ArgumentParser(description="Sets everything above water to no data value")
    parser.add_argument("input_raster", type=str, help="The input raster")
    parser.add_argument("output_raster", type=str, help="The output raster ")
    parser.add_argument("--depth-positive", action="store_true", help="The scratch dir")
    parser.add_argument("--margin", type=float, default=0.0, help="Removes everything below this value. Allows for space for robot to traverse. E.g. use -1.5 (m) to allow for robots that need 1.5m clearance.")
    args = parser.parse_args()
    return args


def main(args):
    if args.input_raster == args.output_raster:
        raise ValueError("Input Raster and Output Raster cannot be the same!")

    input_ds = gdal.Open(args.input_raster)

    no_data_value = input_ds.GetRasterBand(1).GetNoDataValue()
    arr = input_ds.GetRasterBand(1).ReadAsArray()
    if args.depth_positive:
        arr[arr <= args.margin] = no_data_value
    else:
        arr[arr >= args.margin] = no_data_value

    output_ds = gdal.GetDriverByName('GTiff').Create(args.output_raster, input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Float32)
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    output_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    output_ds.FlushCache()

if __name__ == "__main__":
    main(get_args())