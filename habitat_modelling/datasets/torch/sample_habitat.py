import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import plyfile
import numpy as np
import utm
import math
import warnings
import random
import pymap3d
import shapely
import shapely.geometry
from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros, process_depth_map_to_array
try:
    from habitat_modelling.utils.raster_utils import extract_raster_patch, retrieve_pixel_coords, retrieve_geo_coords, extract_geotransform, get_lla_from_pixels, get_pixels_from_lla
except ImportError:
    warnings.warn("Cannot import habitat_modelling.utils.raster_utils (gdal?). Live raster loading not supported")

def get_bounds(info, transform):
    maxx = info['size'][0]
    maxy = info['size'][1]
    ul = retrieve_geo_coords((0,0), transform)
    ur = retrieve_geo_coords((maxx, 0), transform)
    bl = retrieve_geo_coords((0,maxy), transform)
    br = retrieve_geo_coords((maxx,maxy), transform)
    xp = [i[0] for i in [ul, ur, bl, br]]
    yp = [i[1] for i in [ul, ur, bl, br]]
    return min(xp), min(yp), max(xp), max(yp)

class SampleRaster(data.Dataset):
    def __init__(self, rasters=None, raster_paths=None, raster_sizes=None, raster_boundaries=None, raster_transforms=None, length=None):
        """
        Initialises the generator

        Args:
            coco_path: (str) Path to the coco dataset
            rasters: (list) List of strings for the rasters to load. Must match elements of coco dictionary. E.g. ['bathy','backscatter']
            live_raster: (bool) Whether to extract the raster patches online. If not, the patches need to be extracted before (using dive-to-coco)
            raster_sizes: (list) [Only if live_raster] The size of the raster patches to extract
            raster_transforms: (dict) Dictionary mapping each raster to the transform to be applied.
            raster_boundaries: (dict) Dictionary mapping each raster to the transform to be applied.
            length: (dict) Number of samples to extract for each epoch
        """
        self.raster_collection = {}
        self.rasters = rasters

        # Create the raster dictionary
        if len(rasters) != len(raster_paths) or len(rasters) != len(raster_sizes):
            raise ValueError("Length of raster lists need to be the same")

        for n, rs in enumerate(self.rasters):
            raster_ds, geotransform, raster_info = extract_geotransform(raster_paths[n])
            nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()

            if raster_boundaries is not None and rs in raster_boundaries:
                boundary = raster_boundaries[rs]
            else:
                boundary = shapely.geometry.box(*get_bounds(raster_info, geotransform))


            self.raster_collection[rs] = {
                "dataset": raster_ds,
                "info": raster_info,
                "geotransform": geotransform,
                "size": raster_sizes[n],
                "no_data_val": nodataval,
                "boundary": boundary
            }
        # Bathy Transforms
        self.raster_transforms = raster_transforms  # A list of classes with a 'transform' function

        self.length = length if length else None

    def __len__(self):
        """
        Denotes the number of images in each epoch

        Returns:
            (int): number of batches

        """
        return self.length

    def load_raster(self, raster):
        """
        Loads the raster corresponding to the image
        Args:
            raster: (str) The raster format to use
        Returns:
            np.ndarray: The raster at this point.
        """
        boundary = self.raster_collection[raster]['boundary']

        minx, miny, maxx, maxy = boundary.bounds

        valid_point = False
        while not valid_point:
            xbase = np.random.uniform(minx,maxx)
            ybase = np.random.uniform(miny, maxy)
            if boundary.contains(shapely.geometry.Point((xbase,ybase))):
                valid_point = True


        px, py = retrieve_pixel_coords([xbase, ybase], list(self.raster_collection[raster]['geotransform']))


        patch_size = self.raster_collection[raster]['size']

        half_patch = np.floor(patch_size[0]/2)

        off_x = int(np.round(px - half_patch))
        off_y = int(np.round(py - half_patch))

        patch = extract_raster_patch(self.raster_collection[raster]['dataset'], off_x, off_y, patch_size[0])

        # Check for no data values
        # if np.any(patch == self.no_data_values[raster]):
        #     patch = None  # TODO make it work
        if np.any(patch == self.raster_collection[raster]["no_data_val"]):
            patch = None

        if raster == "bathy" or raster == "bathymetry":  # Separate this, as the depth needs to be separated
            if patch is None:
                return None, None
            centre_depth = patch[round(patch.shape[0]/2), round(patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            return dmap, mean_depth
        else:
            return patch

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        data_collected = False

        # Spin until data from each source is detected
        while not data_collected:
            # Load each raster and allocate them to the dictionary
            raster_patches = {}
            for r in self.rasters:
                raster_patches[r] = self.load_raster(raster=r)


            # Check rasters
            bad_raster = False # TODO remove this, temporary
            rasters_ok = True  # This will be set False if any raster is not OK.

            for r,patch in raster_patches.items():
                if r == 'bathy' or r == 'bathymetry':
                    if patch[0] is None or patch[1] is None:
                        dm = np.zeros(self.raster_collection[r]["size"])  # TODO remove this
                        depth = np.array([0.0])  # TODO remove this
                        bad_raster=True # TODO remove this
                        raster_patches[r] = (dm, depth)
                        rasters_ok = False
                    elif np.isnan(patch[1]) or np.isnan(patch[0]).any():
                        dm = np.nan_to_num(patch[0], copy=True)
                        dep = np.nan_to_num(patch[1], copy=True)
                        raster_patches[r] = (dm, dep)
                        rasters_ok = True
                else:
                    if patch is None:
                        rasters_ok = False
                        continue
                    if np.isnan(patch).any():
                        rasters_ok = False

            # Transform the raster patches
            if self.raster_transforms and rasters_ok:
                for r, patch in raster_patches.items():
                    if patch is None:
                        continue
                    if r == 'bathy' or r == 'bathymetry':
                        if patch[0] is None or patch[1] is None:
                            continue
                        dmap = patch[0] # In this case patch is a tuple (dmap, depth)
                        # Only operate on the depth map
                        dmap = self.raster_transforms[r](dmap)
                        # Add the patch tuple back in
                        raster_patches[r] = (dmap, patch[1])
                    else:
                        # Transform the patch
                        patch = self.raster_transforms[r](patch)
                        # Transform the patch
                        raster_patches[r] = patch


            if rasters_ok:
                data_collected = True

        return_list = []

        for r, patch in raster_patches.items():
            if r == 'bathy' or r == 'bathymetry':
                dmap = patch[0]
                depth = patch[1]
                if len(dmap.shape) == 2:
                    dmap = dmap.unsqueeze(0)
                return_list.append(dmap)
                return_list.append(depth)
            else:
                if len(patch.shape) == 2:
                    patch = patch.unsqueeze(0)
                return_list.append(patch)

        return return_list

    def get_output_format(self):
        print_list = []
        for r in self.rasters:
            if r == 'bathy' or r == 'bathymetry':
                print_list.append('dmap')
                print_list.append('depth')
            else:
                print_list.append(r)
        return print_list

    def print_output_format(self):
        print_list = self.get_output_format()
        print(','.join(print_list))

