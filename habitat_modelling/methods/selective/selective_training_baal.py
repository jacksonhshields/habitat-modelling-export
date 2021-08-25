#!/usr/bin/env python3
import argparse
import os
import subprocess
import numpy as np
import json
import pickle
import glob
import pandas as pd
import warnings
import time
import copy
import random
import csv
import yaml
import tqdm

import torch


from habitat_modelling.methods.latent_mapping.geo_inference import get_rasters, get_extractors, to_float_tensor
from habitat_modelling.utils.raster_utils import retrieve_pixel_coords, extract_raster_patch, extract_geotransform, get_ranges, retrieve_geo_coords
from habitat_modelling.utils.raster_utils import get_lla_from_pixels, get_pixels_from_lla, get_bathy_patch, get_raster_patch
from habitat_modelling.methods.latent_mapping.utils import load_latent_model

from pycocotools.coco import COCO

from baal.active.heuristics import BALD, BatchBALD, Variance, Certainty, Random

from habitat_modelling.methods.latent_mapping.train_bayesnn import main as train_bayesnn
from habitat_modelling.methods.latent_mapping.train_drop import main as train_drop
from habitat_modelling.methods.latent_mapping.train_gpc import main as train_gpc


def get_args():
    parser = argparse.ArgumentParser(description="Uses BAAL to heuristics to aid in Active Selection")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default="scratch/",
                        help="Directory to hold all scratch models, logs etc.")
    parser.add_argument('--cuda', action="store_true", help="Whether to use cuda")
    parser.add_argument('--neptune', type=str,
                        help="The neptune experiment name. Omitting this means neptune isn't used.")

    args = parser.parse_args()
    return args

def copy_file(from_fp, to_fp):
    """
    Performs an os copy.

    Args:
        from_fp: (str) the filepath to copy from
        to_fp: (str) the filepath to copy to

    Returns:
        None
    """
    subprocess.check_output(["cp", from_fp, to_fp])

def args_to_cfg(args):
    if args.config.endswith(('.yaml','.yml')):
        cfg = yaml.load(open(args.config, 'r'))
    else:
        cfg = json.load(open(args.config,'r'))

    cfg['run'] = {
        "scratch_dir": args.scratch_dir,
        "neptune": args.neptune,
        "cuda": args.cuda
    }
    return cfg


def subset_coco_from_iids(coco, iids):
    """
    Creates a subset of the complete coco by only choosing the given iids
    Args:
        coco: (pycocotools.coco.COCO) The coco dataset to subset
        iids: (list[int]) A list of image ids to select

    Returns:

    """
    out_coco = copy.deepcopy(coco.dataset)

    images = coco.loadImgs(ids=iids)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=iids))

    out_coco['images'] = images
    out_coco['annotations'] = annotations

    return out_coco


def create_pool_dataset(all_coco, used_iids, save_path):
    """
    Creates a pool dataset by excluding all the iids given. Saves this dataset to file.

    Args:
        all_coco: (pycocotools.coco.COCO) The coco dataset to subset
        used_iids: (list[int]) A list of image ids to exclude
        save_path: (str) The path to save to

    Returns:

    """
    all_iids = all_coco.getImgIds()
    pool_coco = copy.deepcopy(all_coco.dataset)
    pool_iids = []
    for iid in all_iids:
        if iid not in used_iids:
            pool_iids.append(iid)
    images = all_coco.loadImgs(ids=pool_iids)
    annotations = all_coco.loadAnns(all_coco.getAnnIds(imgIds=pool_iids))
    pool_coco['images'] = images
    pool_coco['annotations'] = annotations
    json.dump(pool_coco, open(save_path, 'w'))
    if len(pool_iids) == 0:
        return True
    else:
        return False



def get_baal_heuristic(method, params):
    """
    Creates the bald heuristic to be used
    Args:
        method: (str) The bald heuristic to be used, one of {"BALD", "BatchBALD", "Variance"}
        params: (dict) The dictionary arguments to configure the heuristic

    Returns:

    """
    if method == "BALD":
        heuristic = BALD(**params)
    elif method == "BatchBALD":
        heuristic = BatchBALD(**params)
    elif method == "Variance":
        heuristic = Variance(**params)
    elif method == "Random":
        heuristic = Random(**params)
    else:
        raise ValueError("Method %s not supported" % method)
    return heuristic


def get_inference_model_cfg(iteration_dir, best_model_idx, model_type):
    """
    Creates the model cfg to be used for inference, by looking at the model type and the best model idx (or using the symlinks).
    Args:
        iteration_dir: (str) The path to the iteration scratch directory
        best_model_idx: (int) The epoch of the best model
        model_type: (str) model type

    Returns:

    """
    if 'drop' in model_type:
        model_cfg = {
            "type": model_type,
            "params": os.path.join(iteration_dir, "models", 'model_params.json'),
            "weights": os.path.join(iteration_dir, "models", 'latent_model_best.pt')
        }
    else:  # TODO also do gp
        model_cfg = {
            "type": model_type,
            "params": os.path.join(iteration_dir, "models", 'model_params.json'),
            "weights": os.path.join(iteration_dir, "models", 'bnn_best.pt'),
            "pyro_params": os.path.join(iteration_dir, "models", 'pyro_params_best.pt')
        }
    return model_cfg


def predict(coco_path, model_cfg, iid_to_feature, preprocess_scaler=None, monte_carlo_samples=32, cuda_latent=True):
    """
    Uses the model to predict on the pool data
    Args:
        coco_path: (str) Path to the coco dataset
        model_cfg: (dict) Configures the model
        iid_to_feature: (dict) k=iid, v=feature. Image id to encoded feature
        preprocess_scaler: (str) The path to the preprocess scaler to use. Only used for GP
        monte_carlo_samples: (int) The number of MC samples to use
        cuda_latent: (bool) Whether to use cuda for the prob model.

    Returns:

    """
    coco = COCO(coco_path)
    CUDA_LATENT =cuda_latent
    latent_model = load_latent_model(model_cfg, CUDA=CUDA_LATENT)
    if cuda_latent:
        try:
            latent_model.cuda()
        except AttributeError:
            pass
    if preprocess_scaler is not None:
        preprocess_scaler = pickle.load(open(preprocess_scaler, 'rb'))
    else:
        preprocess_scaler = None

    iid_to_preds = {}

    for image in tqdm.tqdm(coco.loadImgs(coco.getImgIds())):
        feature = iid_to_feature[image['id']]
        if feature is None:
            continue
        feature_t = to_float_tensor(feature).unsqueeze(0)
        if not CUDA_LATENT:
            feature_t = feature_t.type(torch.FloatTensor).cpu()
        else:
            feature_t = feature_t.type(torch.cuda.FloatTensor)

        y_pred = latent_model.predict(feature_t, monte_carlo_samples)
        y_pred = y_pred.squeeze()
        y_pred = np.exp(y_pred)
        iid_to_preds[image['id']] = y_pred

    return iid_to_preds

def train(model_type, cfg):
    """
    Trains the models
    Args:
        cfg: (dict) The configuration for the training. See train_bayesnn.py/train_drop.py/train_gps.py and corresponding cfg for examples.

    Returns:

    """
    if model_type == "bayesnn":
        train_bayesnn(cfg)
    elif model_type == "drop":
        train_drop(cfg)
    elif model_type == "gpc":
        train_gpc(cfg)


def select(selection_method, selection_params, iid_to_preds, num_selections):
    """
    Uses the heuristic to select what samples to use next
    Args:
        selection_method: (str) the selection method to use, one of {"BALD", "BatchBALD", "Variance"}
        selection_params: (dict) the dictionary arguments to configure the baal heuristic.
        iid_to_preds: (dict) k=iid, v=pred. The predictions
        num_selections: (int) The number of new samples to select.

    Returns:

    """
    all_iids = []
    predictions = []
    for k,v in iid_to_preds.items():
        all_iids.append(k)
        predictions.append(v)
    predictions = np.asarray(predictions)
    # TODO create predictions in baal format from iid_to_preds

    # TODO need to create iid list to match

    heuristic = get_baal_heuristic(method=selection_method, params=selection_params)

    # predictions = [n_sample, n_class, ..., n_iterations]

    selected = heuristic(predictions.swapaxes(1,2))[:num_selections]

    selected_iids = [all_iids[idx] for idx in selected]

    return selected_iids

def clean(directory):
    # Read validation logs csv
    valdf = pd.read_csv(os.path.join(directory, 'logs', 'validation_logs.csv'))

    best_model_idx = valdf['Val_Acc'].idxmax()

    all_files = glob.glob(os.path.join(directory, 'models/*'))

    for file in all_files:
        if '.json' in file:
            continue
        elif '_' + str(best_model_idx) in file:
            continue
        else:
            rm_cmd = ["rm", file]
            subprocess.check_output(rm_cmd)

    # Link best dataset
    if os.path.isfile(os.path.join(os.getcwd(), directory, 'models', 'latent_model_%d.pt' %best_model_idx)):
        link_cmd = ["ln", "-s", os.path.join(os.getcwd(), directory, 'models', 'latent_model_%d.pt' % best_model_idx),
                    os.path.join(os.getcwd(), directory, 'models', 'latent_model_best.pt')]
        subprocess.check_output(link_cmd)
    if os.path.isfile(os.path.join(os.getcwd(), directory, 'models', 'bnn_%d.pt' %best_model_idx)):
        link_cmd = ["ln", "-s", os.path.join(os.getcwd(), directory, 'models', 'bnn_%d.pt' %best_model_idx), os.path.join(os.getcwd(), directory, 'models', 'bnn_best.pt')]
        subprocess.check_output(link_cmd)
    if os.path.isfile(os.path.join(os.getcwd(), directory, 'models', 'pyro_params_%d.pt' %best_model_idx)):
        link_cmd = ["ln", "-s", os.path.join(os.getcwd(), directory, 'models', 'pyro_params_%d.pt' %best_model_idx), os.path.join(os.getcwd(), directory, 'models', 'pyro_params_best.pt')]
        subprocess.check_output(link_cmd)

    return best_model_idx


def extract_features(coco_path, raster_config, extractor_config, pickle_path=None, include_depth=True):
    coco = COCO(coco_path)
    raster_registry = get_rasters(raster_config)
    extractors = get_extractors(extractor_config, CUDA=True)
    extractor = extractors[0]
    CUDA = True
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    total_images = len(coco.getImgIds())

    iid_to_feature = {}

    for image in tqdm.tqdm(coco.loadImgs(coco.getImgIds())):
        rs = "bathymetry"
        lla = image['geo_location']
        px, py = get_pixels_from_lla(lla, raster_registry[rs]["geotransform"],
                                     raster_registry[rs]["info"]["projection"])
        half_patch = np.floor(raster_registry[rs]["size"][0] / 2)
        offx = int(np.floor(px - half_patch))
        offy = int(np.floor(py - half_patch))
        if 'bathy' in rs:
            patch, depth = get_bathy_patch(raster_registry[rs]["dataset"], offx, offy, raster_registry[rs]["size"],
                                           raster_registry[rs]["no_data_val"])
        else:
            patch = get_raster_patch(raster_registry[rs]["dataset"], offx, offy, raster_registry[rs]["size"],
                                     raster_registry[rs]["no_data_val"])
        if patch is None:
            continue
        patch_t = to_float_tensor(patch).unsqueeze(0).unsqueeze(0)
        patch_t = patch_t.type(FloatTensor)

        encoded = extractor['model'].predict(patch_t)
        encoded = encoded.detach().cpu().numpy().squeeze(0)
        if include_depth:
            depth = np.expand_dims(np.array(depth), axis=0)
            feature = np.concatenate([encoded, depth], axis=-1)
        else:
            feature = encoded
        iid_to_feature[image['id']] = feature

    if pickle_path is not None:
        pickle.dump(iid_to_feature, open(pickle_path, 'wb'))
    return iid_to_feature

def main(cfg):
    scratch_dir = cfg['run']['scratch_dir']

    # Initialise

    iteration = 0
    max_iteration = cfg['selective'].get('max_iteration', 1e6)
    all_data_used = False

    iid_to_features_pkl = os.path.join(scratch_dir, 'iid_to_features.pkl')
    if os.path.isfile(iid_to_features_pkl):
        iid_to_features = pickle.load(open(iid_to_features_pkl, 'rb'))
    else:
        iid_to_features = extract_features(coco_path=cfg['datasets']['train']['path'],
                                           raster_config=cfg['rasters'],
                                           extractor_config=cfg['extractors'])
        pickle.dump(iid_to_features, open(iid_to_features_pkl, 'wb'))


    iteration_dir = os.path.join(scratch_dir, "iteration_%d" % iteration)
    os.makedirs(iteration_dir, exist_ok=True)
    os.makedirs(os.path.join(iteration_dir, 'datasets'), exist_ok=True)


    all_train_coco = COCO(cfg['datasets']['train']['path'])


    seed_coco = COCO(cfg['datasets']['seed']['path'])
    all_train_iids = seed_coco.getImgIds()

    copy_file(cfg['datasets']['seed']['path'], os.path.join(iteration_dir, 'datasets', 'train.json'))
    copy_file(cfg['datasets']['val']['path'], os.path.join(iteration_dir, 'datasets', 'val.json'))

    # Create the pool dataset
    all_data_used = create_pool_dataset(all_train_coco, all_train_iids,
                                        os.path.join(iteration_dir, "datasets", "pool.json"))

    del seed_coco

    while iteration < max_iteration and all_data_used is False:
        iteration_dir = os.path.join(scratch_dir, "iteration_%d" % iteration)
        # TODO balance datasets???
        dataset_cfg = {
            "train": {"path": os.path.join(iteration_dir, "datasets", "train.json")},
            "val": {"path": os.path.join(iteration_dir, "datasets", "val.json")},
        }

        train_cfg = {
            "training": cfg['training'],
            "model": cfg['model'],
            "rasters": cfg['rasters'],
            "extractors": cfg['extractors'],
            "datasets": dataset_cfg,
            'run': {
                "scratch_dir": iteration_dir,
                "cuda": cfg['run']['cuda'],
                "neptune": cfg['run']['neptune'],
                "cache_dir": os.path.join(iteration_dir, 'cache')}
            }

        # Train
        train(cfg['selective']['model_type'], train_cfg)

        # Clean
        best_model_idx = clean(iteration_dir)
        model_cfg = get_inference_model_cfg(iteration_dir, best_model_idx, cfg['model']['type'])
        # predict
        iid_to_pred = predict(coco_path=os.path.join(iteration_dir, "datasets", "pool.json"),
                              model_cfg=model_cfg,
                              iid_to_feature=iid_to_features,
                              preprocess_scaler=cfg['inference'].get('preprocess_scaler', None),
                              monte_carlo_samples=cfg['inference'].get('monte_carlo_samples', None),
                              cuda_latent=cfg['inference'].get('cuda_latent', False)
                              )

        # Select
        selected_ids = select(cfg['selective']['method'], cfg['selective']['baal_params'], iid_to_pred, cfg['selective']['samples_per_iteration'])

        # Make the next iteration directory
        next_iteration_dir = os.path.join(scratch_dir, "iteration_%d" % (iteration+1))
        os.makedirs(next_iteration_dir, exist_ok=True)
        os.makedirs(os.path.join(next_iteration_dir, 'datasets'), exist_ok=True)

        # Create a coco to summarise the selected datasets
        selected_coco_dict = subset_coco_from_iids(all_train_coco, selected_ids)
        json.dump(selected_coco_dict, open(os.path.join(iteration_dir, "datasets", "selected.json"), 'w'))


        # Create the next training coco
        all_train_iids.extend(selected_ids)
        next_train_coco_dict = subset_coco_from_iids(all_train_coco, all_train_iids)
        json.dump(next_train_coco_dict, open(os.path.join(next_iteration_dir, "datasets", "train.json"), 'w'))

        # Create the pool dataset
        all_data_used = create_pool_dataset(all_train_coco, all_train_iids,
                                            os.path.join(next_iteration_dir, "datasets", "pool.json"))
        # Copy the next validation coco
        copy_file(cfg['datasets']['val']['path'], os.path.join(next_iteration_dir, 'datasets', 'val.json'))

        iteration += 1






if __name__ == "__main__":
    main(args_to_cfg(get_args()))