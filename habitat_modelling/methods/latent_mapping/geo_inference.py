#!/usr/bin/env python3

import json
import yaml
import numpy as np
import os
import torch
import copy
import cv2
from colorama import Fore, Back, Style
import torchvision

try:
    from osgeo import gdal
except ImportError:
    import gdal

from osgeo import osr
import utm
import argparse
from PIL import Image
import pickle
import pyro
from torchvision.transforms import ToPILImage, ToTensor
import warnings

from habitat_modelling.utils.raster_utils import retrieve_pixel_coords, extract_raster_patch, extract_geotransform, get_ranges, retrieve_geo_coords
from habitat_modelling.utils.raster_utils import get_lla_from_pixels, get_pixels_from_lla, get_bathy_patch, get_raster_patch

from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros
from habitat_modelling.utils.display_utils import colour_bathy_patch
from tqdm import tqdm

from scipy.special import softmax

from habitat_modelling.methods.latent_mapping.utils import load_latent_model

enable_qgis = True
try:
    from habitat_modelling.utils.qgis_utils import create_qgis_project, create_qgis_project_from_inference_folder
except ImportError:
    warnings.warn("QGIS import failed")
    enable_qgis = False


def calc_entropy(preds):
    """
    Calculates the entropy of the predictions
    Args:
        preds: (np.ndarray) NxC numpy array

    Returns:
        float: Average entropy

    """
    sum_entropy = 0
    for n in range(preds.shape[0]):
        pred = preds[n]
        # pred = softmax(pred)  # TODO temporary
        pred = pred / np.sum(pred)
        ent = -np.sum(pred*np.log(pred))
        sum_entropy += ent
    return sum_entropy / preds.shape[0]


def get_mean_prob_bnn(y_pred):
    """
    Gets the mean and probability of each class given an array of monte carlo predictions. Assumes there was a softplus activation (coming out of a Bayesian Neural Network).

    Args:
        y_pred: (np.ndarray) monte carlo predictions (NxC)
    Returns:
        mean: (np.ndarray) the mean per class activation (C,)
        prob: (np.ndarray) the probability per class (C,)
    """
    y_mean = np.mean(np.exp(y_pred), axis=0)

    # --------------------
    # Get probabilities
    # --------------------
    probs = []
    for cl in range(y_pred.shape[1]):
        histo = []
        histo_exp = []
        for z in range(y_pred.shape[0]):
            histo.append(y_pred[z][cl])
            histo_exp.append(np.exp(y_pred[z][cl]))
        prob = np.percentile(histo_exp, 50)  # sample median probability
        probs.append(prob)
    probs = np.asarray(probs)
    return y_mean, probs


def get_mean_prob_bnn(y_pred):
    """
    Gets the mean and probability of each class given an array of monte carlo predictions.
    Assumes there was a softplus activation (coming out of a Bayesian Neural Network).

    Args:
        y_pred: (np.ndarray) monte carlo predictions (NxC)
    Returns:
        mean: (np.ndarray) the mean per class activation (C,)
        prob: (np.ndarray) the probability per class (C,)
    """
    y_mean = np.mean(np.exp(y_pred), axis=0)

    # --------------------
    # Get probabilities
    # --------------------
    probs = []
    for cl in range(y_pred.shape[1]):
        histo = []
        histo_exp = []
        for z in range(y_pred.shape[0]):
            histo.append(y_pred[z][cl])
            histo_exp.append(np.exp(y_pred[z][cl]))
        prob = np.percentile(histo_exp, 50)  # sample median probability
        probs.append(prob)
    probs = np.asarray(probs)
    return y_mean, probs


def get_mean_prob_dense(y_pred):
    """
    Returns the mean and probability of each class given an array of monte carlo predictions.
    Assumes there was a softmax activation (coming out of a standard neural network).

    Args:
        y_pred: (np.ndarray) monte carlo predictions (NxC)

    Returns:
        mean: (np.ndarray) the mean per class activation (C,)
        prob: (np.ndarray) the probability per class (C,)

    """
    # TODO atm just outputs mean, variance. Make it output probability.
    return np.mean(y_pred, axis=0), np.var(y_pred, axis=0)


def get_mean_var_uncertainty_aleatoric_epistemic(y_pred, output_activation='softmax'):
    """ Gets the mean, var, uncertainty (total), aleatoric and epistemic uncertainty from the predictions.
    softmax: https://gitlab.com/wdeback/dl-keras-tutorial/blob/master/notebooks/3-cnn-segment-retina-uncertainty.ipynb
    softplus: https://github.com/kumar-shridhar/PyTorch-Softplus-Normalization-Uncertainty-Estimation-Bayesian-CNN/blob/master/main.ipynb

    Args:
        y_pred (type): Description of parameter `y_pred`.
        output_activation (type): Description of parameter `output_activation`.

    Returns:
        type: Description of returned object.

    """

    if output_activation == 'softmax':
        y_mean = np.mean(y_pred, axis=0)
        y_var = np.var(y_pred, axis=0)
        aleatoric = np.mean(y_pred*(1 - y_pred), axis=0)
        epistemic = np.mean(y_pred**2, axis=0) - np.mean(y_pred, axis=0)**2
    elif output_activation == 'log_softmax':
        y_pred = np.exp(y_pred)
        y_mean = np.mean(y_pred, axis=0)
        y_var = np.var(y_pred, axis=0)
        aleatoric = np.mean(y_pred * (1 - y_pred), axis=0)
        epistemic = np.mean(y_pred ** 2, axis=0) - np.mean(y_pred, axis=0) ** 2
    elif output_activation == 'softplus':
        # y_pred = np.exp(y_pred)  # No EXP
        # Nornalize
        y_pred = y_pred/np.sum(y_pred).reshape(-1, 1)
        y_mean = np.mean(y_pred, axis=0)
        y_var = np.var(y_pred, axis=0)
        res = np.mean(y_pred, axis=0)
        aleatoric = np.diag(res) - y_pred.T.dot(y_pred)/y_pred.shape[0]
        tmp = y_pred - res
        epistemic = tmp.T.dot(tmp)/tmp.shape[0]
        aleatoric = np.sum(aleatoric, keepdims=True)
        epistemic = np.sum(epistemic, keepdims=True)
    else:
        raise ValueError("Only softplus and softmax activations are supported for calculating uncertainties")
    aleatoric = np.sum(np.squeeze(aleatoric))
    epistemic = np.sum(np.squeeze(epistemic))
    # uncertainty = aleatoric + epistemic
    uncertainty = np.sum(y_var)
    #print("Aleatoric", aleatoric)
    #print("Epistemic", epistemic)
    return y_mean, y_var, uncertainty, aleatoric, epistemic


def to_float_tensor(x):
    if x.dtype == np.uint8:  # Normalise if an integer
        x = x/255
    return torch.FloatTensor(x)


def create_qgis_project(output_dir):
    raise NotImplementedError("TODO")

def args_to_cfg(args):
    if args.config.endswith(('.yaml','.yml')):
        cfg = yaml.load(open(args.config, 'r'))
    else:
        cfg = json.load(open(args.config,'r'))

    cfg['run'] = {
        "scratch_dir": args.scratch_dir,
        "cuda": args.cuda,
        "cuda_latent": args.cuda_latent,
        "qgis_export": args.qgis_export
    }
    return cfg

def get_args():
    parser = argparse.ArgumentParser(description="Trains a multimodal (image + bathy) autoencoder")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default=".",
                        help="Directory to hold all scratch models, logs etc.")
    parser.add_argument('--cuda', action="store_true", help="Whether to use cuda")
    parser.add_argument('--cuda-latent', action="store_true", help="Whether to use cuda")
    parser.add_argument('--qgis-export', action="store_true", help="Whether to export qgis project")
    args = parser.parse_args()
    if args.cuda and not args.cuda_latent:
        warnings.warn("CUDA is selected but cuda latent is not!")
    return args

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
            "no_data_val": nodataval
        }
    return raster_registry


def main(cfg):
    CUDA = cfg['run']['cuda']
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    if CUDA and cfg['run']['cuda_latent']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # TODO - hacky. This is to make pyro guide go on GPU.

    # ----------------
    # Check params
    # ---------------
    sample_rate = cfg['inference'].get('sample_rate')
    if sample_rate > 1.0 or sample_rate < 0.0:
        raise ValueError("Sample ratio should be between 0 and 1")

    x_range = cfg['inference'].get('x_range', None)
    y_range = cfg['inference'].get('y_range', None)


    verbose = cfg['run'].get('verbose', False)

    # -----------------------------
    # Get Encoders
    # -----------------------------
    extractors = get_extractors(cfg['extractors'], CUDA)

    # ------------------------------
    # Get Rasters
    # ------------------------------
    raster_registry = get_rasters(cfg['rasters'])

    # Select a leader raster - used as the basis for geotransforms, ranges, etc.
    leading_raster = raster_registry[list(raster_registry.keys())[0]]
    geotransform = leading_raster["geotransform"]
    ds_info = leading_raster["info"]

    x_samples, y_samples, num_x_samples, num_y_samples, x_range, y_range = get_ranges(x_range, y_range, sample_rate, geotransform, ds_info)

    # This in in pixels!
    dist_between_samples = np.abs(x_samples[1] - x_samples[0])  # pixels

    # Calculate new pixel sizes
    pxl_width = geotransform[1] * (x_samples[-1] - x_samples[0])/num_x_samples
    pxl_height = geotransform[5] * (y_samples[-1] - y_samples[0])/num_y_samples

    latent_model = load_latent_model(cfg['model'], cfg['run']['cuda_latent'])
    if CUDA and cfg['run']['cuda_latent']:
        try:
            latent_model.cuda()
        except AttributeError:
            warnings.warn("Couldn't move latent model to CUDA")
    if cfg['inference'].get('preprocess_scaler', None):
        preprocess_scaler = pickle.load(open(cfg['inference']['preprocess_scaler'], 'rb'))
    else:
        preprocess_scaler = None

    results = {
        "ij": [],
        "most_likely": [],
        "mean": [],
        "prob": [],
        "epistemic": [],
        "aleatoric": [],
        "var": [],
        "uncertainty": []
    }
    count = 0
    total_samples = num_x_samples*num_y_samples
    pbar = tqdm(total=total_samples, desc="Inferring")

    for i in range(num_x_samples):
        for j in range(num_y_samples):
            x_base = x_samples[i]
            y_base = y_samples[j]
            lla = get_lla_from_pixels((x_base, y_base), leading_raster['geotransform'],
                                      leading_raster['info']['projection'])
            data_ok = True

            features = []
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
                        patch = get_raster_patch(raster_registry[rs]["dataset"], offx, offy, raster_registry[rs]["size"], raster_registry[rs]["no_data_val"])
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
                encoded = extractor['model'].predict(input_t)
                features.append(encoded.detach().cpu().numpy().squeeze(0))
                if depth:
                    depth = np.expand_dims(np.array(depth), axis=0)
                    features.append(depth)

            if len(features) > 0:
                encoded = np.concatenate(features, axis=-1)
                if preprocess_scaler:
                    encoded = preprocess_scaler.transform(np.expand_dims(encoded, 0))
                    encoded = encoded.squeeze(0)
            else:
                data_ok = False

            if data_ok:
                encoded_t = torch.tensor(encoded).unsqueeze(0)
                if not cfg['run']['cuda_latent']:
                    encoded_t = encoded_t.type(torch.FloatTensor).cpu()
                else:
                    encoded_t = encoded_t.type(FloatTensor)

                y_pred = latent_model.predict(encoded_t, cfg['inference'].get('monte_carlo_samples', 32))
                y_pred = y_pred.squeeze()

                if 'bnn' in cfg['model']['type']:
                    y_mean, y_var, y_uncertainty, y_aleatoric, y_epistemic = get_mean_var_uncertainty_aleatoric_epistemic(
                        y_pred, output_activation='softplus')
                    y_mean, probs = get_mean_prob_bnn(y_pred)  # Override y_mean with exponential y_mean. Add in probs
                elif 'gp_grid' in cfg['model']['type']:
                    y_mean, y_var, y_uncertainty, y_aleatoric, y_epistemic = get_mean_var_uncertainty_aleatoric_epistemic(
                        y_pred, output_activation='softmax')
                    probs = y_var  # TODO
                else:
                    y_mean, y_var, y_uncertainty, y_aleatoric, y_epistemic = get_mean_var_uncertainty_aleatoric_epistemic(
                        y_pred, output_activation='log_softmax')
                    probs = y_var  # TODO

                results['ij'].append(np.array([i, j]))
                results['mean'].append(y_mean)
                results['prob'].append(probs)
                most_likely_pred = np.argmax(y_mean)
                results['most_likely'].append(most_likely_pred)
                results['var'].append(y_var)
                results['aleatoric'].append(y_aleatoric)
                results['epistemic'].append(y_epistemic)
                results['uncertainty'].append(y_uncertainty)
                if verbose:
                    print("Inferring %d/%d locations" % (count, total_samples))
            else:
                if verbose:
                    print("Skipping %d/%d locations" % (count, total_samples))
            count += 1
            pbar.update(1)
    pbar.close()

    # These should all have the same length! - maybe put them in a dict to highlight this
    ij = np.asarray(results['ij'])
    mean_array = np.asarray(results['mean'])
    prob_array = np.asarray(results['prob'])
    most_likely_array = np.asarray(results['most_likely'])
    aleatoric_array = np.asarray(results['aleatoric'])
    epistemic_array = np.asarray(results['epistemic'])
    var_array = np.asarray(results['var'])
    uncertainty_array = np.asarray(results['uncertainty'])


    # --------------------------------------------------
    # Create the bands to be written to the geotiffs
    # --------------------------------------------------
    category_band = np.zeros([num_y_samples, num_x_samples]) - 1
    mean_bands = np.zeros([num_y_samples, num_x_samples, mean_array.shape[1]]) - 1
    prob_bands = np.zeros([num_y_samples, num_x_samples, prob_array.shape[1]]) - 1
    var_bands = np.zeros([num_y_samples, num_x_samples, var_array.shape[1]]) - 1

    aleatoric_band = np.zeros([num_y_samples, num_x_samples]) - 1
    epistemic_band = np.zeros([num_y_samples, num_x_samples]) - 1
    uncertainty_band = np.zeros([num_y_samples, num_x_samples]) - 1

    # ------------------
    # Fill the bands
    # ------------------
    for n in range(ij.shape[0]):
        i = ij[n,0]
        j = ij[n,1]
        category_band[j,i] = most_likely_array[n]
        mean_bands[j,i,:] = mean_array[n,:]
        prob_bands[j,i,:] = prob_array[n, :]

        var_bands[j,i,:] = var_array[n,:]

        aleatoric_band[j,i] = aleatoric_array[n]
        epistemic_band[j,i] = epistemic_array[n]
        uncertainty_band[j,i] = uncertainty_array[n]

    # -------------------------------
    # Set the new geotransform size
    # ------------------------------
    out_geotransform = list(geotransform)
    out_geotransform[0] = x_range[0]  # TODO how to get starting val
    out_geotransform[3] = y_range[1]  # TODO how to get starting val
    out_geotransform[1] = pxl_width
    out_geotransform[5] = pxl_height

    json.dump(out_geotransform, open(os.path.join(cfg['run']['scratch_dir'], 'out_geotransform.json'),'w'))
    geo_ds = leading_raster["dataset"]
    # ---------------------------
    # Create category geotiff
    # ---------------------------
    category_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "category_map.tif"), num_x_samples, num_y_samples, 1, gdal.GDT_Float32)
    category_ds.SetGeoTransform(out_geotransform)
    category_ds.SetProjection(geo_ds.GetProjection())
    category_ds.GetRasterBand(1).WriteArray(category_band.astype(np.float32))
    category_ds.GetRasterBand(1).SetNoDataValue(-1.0)

    category_ds.FlushCache()  # write to disk
    category_ds = None

    # ---------------------------
    # Create mean geotiff
    # ---------------------------
    mean_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "mean_map.tif"), num_x_samples,
                                                       num_y_samples, mean_bands.shape[2], gdal.GDT_Float32)
    mean_ds.SetGeoTransform(out_geotransform)
    mean_ds.SetProjection(geo_ds.GetProjection())
    for n in range(mean_bands.shape[2]):
        mean_ds.GetRasterBand(n+1).WriteArray(mean_bands[:,:,n].astype(np.float32))
        mean_ds.GetRasterBand(n+1).SetNoDataValue(-1.0)
    mean_ds.FlushCache()  # write to disk
    mean_ds = None


    # ---------------------------
    # Create transparency geotiff
    # ----------------------------
    # Creates a transparency layer, which is equal to the activation of the most active class
    transparency_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "category_transparency.tif"), num_x_samples, num_y_samples, 2, gdal.GDT_Float32)
    transparency_ds.SetGeoTransform(out_geotransform)
    transparency_ds.SetProjection(geo_ds.GetProjection())
    transparency_ds.GetRasterBand(1).WriteArray(category_band.astype(np.float32))
    transparency_ds.GetRasterBand(2).WriteArray(mean_bands.max(axis=-1).astype(np.float32)*100.0)
    transparency_ds.GetRasterBand(1).SetNoDataValue(-1.0)

    transparency_ds.FlushCache()  # write to disk
    transparency_ds = None


    # ---------------------------
    # Create var geotiff
    # ---------------------------
    var_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "var_map.tif"), num_x_samples,
                                                       num_y_samples, var_bands.shape[2], gdal.GDT_Float32)
    var_ds.SetGeoTransform(out_geotransform)
    var_ds.SetProjection(geo_ds.GetProjection())
    for n in range(mean_bands.shape[2]):
        var_ds.GetRasterBand(n+1).WriteArray(var_bands[:,:,n].astype(np.float32))
        var_ds.GetRasterBand(n+1).SetNoDataValue(-1.0)
    var_ds.FlushCache()  # write to disk
    var_ds = None


    # ---------------------------
    # Create entropy geotiff
    # ---------------------------
    xt = mean_bands
    inds = xt == -1
    inds = inds[:,:,0]
    # Remove -1 elements (correspond to invalid)
    xt[xt==-1] = 1e-6
    # Clip to just above 0 to stabilise entropy calc
    xt[xt < 1e-6] = 1e-6
    entropy_band = -np.sum(xt * np.log(xt),axis=-1)
    entropy_band[inds] = -1
    entropy_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "entropy_map.tif"), num_x_samples,
                                                       num_y_samples, 1, gdal.GDT_Float32)
    entropy_ds.SetGeoTransform(out_geotransform)
    entropy_ds.SetProjection(geo_ds.GetProjection())
    entropy_ds.GetRasterBand(1).WriteArray(entropy_band.astype(np.float32))
    entropy_ds.GetRasterBand(1).SetNoDataValue(-1.0)
    entropy_ds.FlushCache()
    entropy_ds = None

    # ---------------------------
    # Create prob geotiff
    # ---------------------------
    prob_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "prob_map.tif"), num_x_samples,
                                                       num_y_samples, prob_bands.shape[2], gdal.GDT_Float32)
    prob_ds.SetGeoTransform(out_geotransform)
    prob_ds.SetProjection(geo_ds.GetProjection())
    for n in range(prob_bands.shape[2]):
        prob_ds.GetRasterBand(n+1).WriteArray(prob_bands[:,:,n].astype(np.float32))
        prob_ds.GetRasterBand(n + 1).SetNoDataValue(-1.0)
    prob_ds.FlushCache()  # write to disk
    prob_ds = None

    # --------------------------
    # Create Uncertainty geotiff
    # --------------------------
    uncertainty_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "uncertainty_map.tif"), num_x_samples, num_y_samples, 1, gdal.GDT_Float32)
    uncertainty_ds.SetGeoTransform(out_geotransform)
    uncertainty_ds.SetProjection(geo_ds.GetProjection())
    uncertainty_ds.GetRasterBand(1).WriteArray(uncertainty_band.astype(np.float32))
    uncertainty_ds.GetRasterBand(1).SetNoDataValue(-1.0)
    uncertainty_ds.FlushCache()
    uncertainty_ds = None

    # --------------------------
    # Create Aleatoric geotiff
    # --------------------------
    aleatoric_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "aleatoric_map.tif"), num_x_samples, num_y_samples, 1, gdal.GDT_Float32)
    aleatoric_ds.SetGeoTransform(out_geotransform)
    aleatoric_ds.SetProjection(geo_ds.GetProjection())
    aleatoric_ds.GetRasterBand(1).WriteArray(aleatoric_band.astype(np.float32))
    aleatoric_ds.GetRasterBand(1).SetNoDataValue(-1.0)
    aleatoric_ds.FlushCache()
    aleatoric_ds = None

    # --------------------------
    # Create epistemic geotiff
    # --------------------------
    epistemic_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "epistemic_map.tif"), num_x_samples, num_y_samples, 1, gdal.GDT_Float32)
    epistemic_ds.SetGeoTransform(out_geotransform)
    epistemic_ds.SetProjection(geo_ds.GetProjection())
    epistemic_ds.GetRasterBand(1).WriteArray(epistemic_band.astype(np.float32))
    epistemic_ds.GetRasterBand(1).SetNoDataValue(-1.0)
    epistemic_ds.FlushCache()
    epistemic_ds = None


    # ---------------------------
    # Create per category geotiff
    # ---------------------------

    for n in range(mean_bands.shape[2]):
        category_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(cfg['run']['scratch_dir'], "category_%d.tif"%n), num_x_samples,
                                                       num_y_samples, 2, gdal.GDT_Float32)
        category_ds.SetGeoTransform(out_geotransform)
        category_ds.SetProjection(geo_ds.GetProjection())
        category_ds.GetRasterBand(1).WriteArray(mean_bands[:, :, n].astype(np.float32))
        category_ds.GetRasterBand(2).WriteArray(var_bands[:, :, n].astype(np.float32))
        category_ds.GetRasterBand(1).SetNoDataValue(-1.0)
        category_ds.GetRasterBand(2).SetNoDataValue(-1.0)
        category_ds.FlushCache()
        category_ds = None
    geo_ds = None

    if enable_qgis and cfg['run'].get("qgis_export", False):
        bathymetry_path = list(cfg['rasters'].values())[0]['path']
        create_qgis_project_from_inference_folder(folder=cfg['run']['scratch_dir'], bathymetry=bathymetry_path)



if __name__ == "__main__":
    main(args_to_cfg(get_args()))
