#!/usr/bin/env python3

import json
import yaml
import numpy as np
import os
import copy
import cv2
from colorama import Fore, Back, Style
import warnings
try:
    from osgeo import gdal
except ImportError:
    import gdal

from osgeo import osr
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import torch
from PIL import Image
import pickle
from tqdm import tqdm

from habitat_modelling.utils.raster_utils import retrieve_pixel_coords, extract_raster_patch, extract_geotransform, get_ranges
from habitat_modelling.utils.raster_utils import get_lla_from_pixels, get_pixels_from_lla, get_bathy_patch, get_raster_patch
from habitat_modelling.utils.general_utils import get_dict_arguments
from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros

from habitat_modelling.methods.latent_mapping.feature_extractors import BathymetryEncoder, BathymetryVariationalEncoder

enable_qgis = True
try:
    from habitat_modelling.utils.qgis_utils import create_qgis_project, create_qgis_project_from_inference_folder
except ImportError:
    warnings.warn("QGIS import failed")
    enable_qgis = False


def args_to_cfg(args):
    if args.config.endswith(('.yaml','.yml')):
        cfg = yaml.load(open(args.config, 'r'))
    else:
        cfg = json.load(open(args.config,'r'))

    cfg['run'] = {
        "scratch_dir": args.scratch_dir,
        "cuda": args.cuda,
        "qgis_export": args.qgis_export
    }
    return cfg

def get_args():
    parser = argparse.ArgumentParser(description="Runs an autoencoder inference on rasters to output feature arrays.")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default=".",
                        help="Directory to hold all scratch models, logs, outputs, etc.")
    parser.add_argument('--cuda', action="store_true", help="Whether to use cuda")
    parser.add_argument('--qgis-export', action="store_true", help="Whether to export qgis project")
    args = parser.parse_args()
    return args

def to_float_tensor(x):
    if x.dtype == np.uint8:  # Normalise if an integer
        x = x/255
    return torch.FloatTensor(x)

def get_extractors(extractor_cfg, CUDA):
    from habitat_modelling.methods.latent_mapping.feature_extractors import load_extractor
    extractors = []
    for ext_cfg in extractor_cfg:
        ext = {
            "inputs": ext_cfg["inputs"],
            "model": load_extractor(ext_cfg, CUDA)
        }
        extractors.append(ext)
    return extractors

def infer_on_rasters(cfg, raster_registry, range_config, verbose=False):
    # ----------------------------
    # Get Encoders
    # ----------------------------
    extractors = get_extractors(cfg['extractors'], CUDA=cfg['run']['cuda'])
    CUDA = cfg['run']['cuda']

    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

    # -----------------------------
    # Setup ranges
    # -----------------------------

    # Select a leader raster - used as the basis for geotransforms, ranges, etc.
    leading_raster = raster_registry[list(raster_registry.keys())[0]]
    geotransform = leading_raster["geotransform"]
    ds_info = leading_raster["info"]


    x_samples = range_config['x_samples']
    y_samples = range_config['y_samples']
    num_x_samples = range_config['num_x_samples']
    num_y_samples = range_config['num_y_samples']
    x_range = range_config['x_range']
    y_range = range_config['y_range']

    encoded_list = []
    var_list = []
    depth_list = []
    ij_list = []

    total_samples = num_x_samples * num_y_samples
    pbar = tqdm(total=total_samples, desc="Inferring")
    count = 0
    for i in range(num_x_samples):
        for j in range(num_y_samples):
            x_base = x_samples[i]
            y_base = y_samples[j]
            lla = get_lla_from_pixels((x_base, y_base), leading_raster['geotransform'],
                                      leading_raster['info']['projection'])
            data_ok = True

            features = []
            feature_vars = []
            for extractor in extractors:
                patches = []
                depth = None
                for rs in extractor['inputs']:
                    px, py = get_pixels_from_lla(lla, raster_registry[rs]["geotransform"], raster_registry[rs]["info"]["projection"])
                    half_patch = np.floor(raster_registry[rs]["size"][0]/2)
                    offx = int(np.floor(px - half_patch))
                    offy = int(np.floor(py - half_patch))
                    if 'bathy' in rs:
                        patch, depth = get_bathy_patch(raster_registry[rs]["dataset"], offx, offy, raster_registry[rs]["size"], raster_registry[rs]["no_data_val"])
                    else:
                        patch = get_bathy_patch(raster_registry[rs]["dataset"], offx, offy, raster_registry[rs]["size"], raster_registry[rs]["no_data_val"])
                    if patch is None:
                        data_ok = False
                        break
                    patch_t = to_float_tensor(patch).unsqueeze(0).unsqueeze(0)
                    patch_t = patch_t.type(FloatTensor)
                    patches.append(patch_t)
                if not data_ok:
                    break
                if len(patches) == 0:
                    input_t = patch_t
                else:
                    input_t = torch.cat(patches, dim=1)
                if isinstance(extractor['model'], BathymetryVariationalEncoder):  # Check if variational
                    zb, z_mu, z_var = extractor['model'].predict(input_t)
                    features.append(z_mu.detach().cpu().numpy())
                    feature_vars.append(z_var.detach().cpu().numpy())
                else:  # do standard autoencoder
                    encoded = extractor['model'].predict(input_t)
                    features.append(encoded.detach().cpu().numpy())
                if depth:
                    depth = np.expand_dims(np.array(depth), axis=0)

            if len(features) > 0:
                encoded = np.concatenate(features, axis=-1)
                if len(feature_vars) > 0:
                    encoded_vars = np.concatenate(feature_vars, axis=-1)
                else:
                    encoded_vars = None
            else:
                data_ok = False

            if data_ok:
                encoded_list.append(encoded.squeeze(0))
                depth_list.append(depth)
                ij_list.append(np.array([i,j]))
                if encoded_vars is not None:
                    var_list.append(encoded_vars.squeeze(0))


                if verbose:
                    print("Inferring %d/%d locations" % (count, total_samples))
            else:
                if verbose:
                    print("Skipping %d/%d locations" % (count, total_samples))
            count += 1
            pbar.update(1)
    pbar.close()
    encoded_array = np.asarray(encoded_list)
    depth_array = np.asarray(depth_list)
    ij_array = np.asarray(ij_list)
    if len(var_list) > 0:
        var_array = np.asarray(var_list)
    else:
        var_array = None

    data = {"encoded": encoded_array, "depth": depth_array, "ij": ij_array, "var": var_array}
    pickle.dump(data, open(os.path.join(cfg['run']['scratch_dir'], 'inference.pkl'), 'wb'))

    return encoded_array, depth_array, ij_array, var_array


def load_output(path):
    data = pickle.load(open(path, 'rb'))
    X = data['encoded']
    depth = data['depth']
    ij = data['ij']
    var_array = data['var']
    return X, depth, ij, var_array


def get_rasters(raster_cfg):
    raster_registry = {}
    for key,entry in raster_cfg.items():
        raster_ds, geotransform, raster_info = extract_geotransform(entry['path'])
        nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()
        raster_registry[key] = {
            "dataset": raster_ds,
            "info": raster_info,
            "geotransform": geotransform,
            "size": entry['size'],
            "no_data_val": nodataval,
            "path": entry['path']
        }
    return raster_registry


def main(cfg):
    scratch_dir = cfg['run']['scratch_dir']
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    # ----------------
    # Check params
    # ---------------
    sample_rate = cfg['inference'].get('sample_rate')
    if sample_rate > 1.0 or sample_rate < 0.0:
        raise ValueError("Sample ratio should be between 0 and 1")

    x_range = cfg['inference'].get('x_range', None)
    y_range = cfg['inference'].get('y_range', None)

    # ------------------------------
    # Get Rasters
    # ------------------------------
    raster_registry = get_rasters(cfg['rasters'])

    # Select a leader raster - used as the basis for geotransforms, ranges, etc.
    leading_raster = raster_registry[list(raster_registry.keys())[0]]
    geotransform = leading_raster["geotransform"]
    ds_info = leading_raster["info"]

    x_samples, y_samples, num_x_samples, num_y_samples, x_range, y_range = get_ranges(x_range, y_range, sample_rate,
                                                                                      geotransform, ds_info)
    range_cfg = {
        "x_samples": x_samples,
        "y_samples": y_samples,
        "num_x_samples": num_x_samples,
        "num_y_samples": num_y_samples,
        "x_range": x_range,
        "y_range": y_range
    }

    # This in in pixels!
    dist_between_samples = np.abs(x_samples[1] - x_samples[0])  # pixels

    # Calculate new pixel sizes
    pxl_width = geotransform[1] * (x_samples[-1] - x_samples[0]) / num_x_samples
    pxl_height = geotransform[5] * (y_samples[-1] - y_samples[0]) / num_y_samples


    if cfg['inference'].get('load_pickle', None):
        # If a pickle has been given, load the output. Means inference doesn't have to be run again
        X, depth, ij, var_array = load_output(cfg['inference']['load_pickle'])
    else:
        # Run inference
        X, depth, ij, var_array = infer_on_rasters(cfg, raster_registry, range_cfg)

    if cfg['inference'].get('just_inference', False):
        exit()

    # Create an output array. This will be resized to become the full size
    feature_band = np.zeros([num_y_samples, num_x_samples, X.shape[-1]]) - 9999.0
    depth_band = np.zeros([num_y_samples, num_x_samples]) - 9999.0
    if var_array is not None:
        var_band = np.zeros([num_y_samples, num_x_samples, var_array.shape[-1]]) - 9999.0


    # Fill the raster band with the predicted labels
    for n in range(ij.shape[0]):
        i = ij[n,0]
        j = ij[n,1]
        feature_band[j,i,:] = X[n,:]
        depth_band[j,i] = depth[n]
        if var_array is not None:
            var_band[j,i,:] = var_array[n,:]
    output_tiff = os.path.join(scratch_dir, 'clusters.tif')

    # -------------------------------
    # Set the new geotransform size
    # ------------------------------
    out_geotransform = list(geotransform)
    out_geotransform[0] = x_range[0]
    out_geotransform[3] = y_range[1]
    out_geotransform[1] = pxl_width
    out_geotransform[5] = pxl_height

    if var_array is not None:  # If it is variational, call the features 'mu'
        feature_path = os.path.join(scratch_dir, "mu.tif")
    else:
        feature_path = os.path.join(scratch_dir, "features.tif")
    feature_ds = gdal.GetDriverByName('GTiff').Create(feature_path, num_x_samples, num_y_samples, feature_band.shape[-1], gdal.GDT_Float32)
    feature_ds.SetGeoTransform(out_geotransform)
    feature_ds.SetProjection(leading_raster["dataset"].GetProjection())
    for ft in range(feature_band.shape[-1]):
        feature_ds.GetRasterBand(ft+1).WriteArray(feature_band[:,:,ft].astype(np.float32))
    feature_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    feature_ds.FlushCache()

    if var_array is not None:
        var_path = os.path.join(scratch_dir, "var.tif")
        var_ds = gdal.GetDriverByName('GTiff').Create(var_path, num_x_samples, num_y_samples,
                                                          var_band.shape[-1], gdal.GDT_Float32)
        var_ds.SetGeoTransform(out_geotransform)
        var_ds.SetProjection(leading_raster["dataset"].GetProjection())
        for ft in range(var_band.shape[-1]):
            var_ds.GetRasterBand(ft + 1).WriteArray(var_band[:, :, ft].astype(np.float32))
        var_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
        var_ds.FlushCache()

    depth_path = os.path.join(scratch_dir, "depth.tif")
    depth_ds = gdal.GetDriverByName('GTiff').Create(depth_path, num_x_samples, num_y_samples, 1, gdal.GDT_Float32)
    depth_ds.SetGeoTransform(out_geotransform)
    depth_ds.SetProjection(leading_raster["dataset"].GetProjection())
    depth_ds.GetRasterBand(1).WriteArray(depth_band.astype(np.float32))
    depth_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    depth_ds.FlushCache()  # write to disk
    depth_ds = None



if __name__ == "__main__":
    main(args_to_cfg(get_args()))
