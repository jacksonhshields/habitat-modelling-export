#!/usr/bin/env python3
import copy
import os
import json
import yaml
import argparse
import torch
import warnings
import pandas as pd
import numpy as np
from habitat_modelling.methods.latent_mapping.utils import create_datasets,  load_latent_model, to_float_tensor
from tqdm import tqdm
from habitat_modelling.datasets.torch.coco_habitat import CocoRaster
from habitat_modelling.methods.latent_mapping.feature_extractors import LatentGenerator
from habitat_modelling.methods.latent_mapping.utils import load_feature_extractor
from torchvision import transforms
from pycocotools.coco import COCO

def get_args():
    parser = argparse.ArgumentParser(description="Evaluates a model")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default="scratch/",
                        help="Directory to hold all scratch models, logs etc.")
    parser.add_argument('--cache-dir', type=str, help="The cache directory")
    parser.add_argument('--cuda', action="store_true", help="Whether to use cuda")
    parser.add_argument('--cuda-latent', action="store_true", help="Whether to use cuda")
    parser.add_argument('--neptune', type=str,
                        help="The neptune experiment name. Omitting this means neptune isn't used.")
    parser.add_argument('--verbose', action="store_true", help="Whether to evaluate ")

    args = parser.parse_args()
    return args

def args_to_cfg(args):
    if args.config.endswith(('.yaml','.yml')):
        cfg = yaml.load(open(args.config, 'r'))
    else:
        cfg = json.load(open(args.config,'r'))

    cfg['run'] = {
        "scratch_dir": args.scratch_dir,
        "neptune": args.neptune,
        "cuda": args.cuda,
        "cuda_latent": args.cuda_latent,
        "cache_dir": args.cache_dir,
        "verbose": args.verbose
    }
    return cfg

def load_dataset(config):
    raster_names = []
    raster_sizes = []
    raster_paths = []
    raster_transforms = {}
    extractors = {}
    if 'extractors' in config:  # RECCOMENDED - this is compatible with geoinference
        from habitat_modelling.methods.latent_mapping.geo_inference import get_extractors

        extractor_list = get_extractors(config['extractors'], config['run']['cuda'])

        for name, entry in config['rasters'].items():
            raster_names.append(name)
            raster_paths.append(entry['path'])
            raster_sizes.append(entry['size'])
            raster_transforms[name] = to_float_tensor
        raster_names = list(config['rasters'].keys())
        for tractor in extractor_list:
            if len(tractor['inputs']) > 1:
                raise ValueError("Only supports one input")
            if tractor['inputs'][0] in raster_names:
                extractors[tractor['inputs'][0]] = tractor['model']
    else:  # BACKWARDS COMPATIBLE with train scripts but not with geoinference
        for name, entry in config['rasters'].items():
            raster_names.append(name)
            raster_paths.append(entry['path'])
            raster_sizes.append(entry['size'])
            raster_transforms[name] = to_float_tensor
            extractors[name] = load_feature_extractor(entry['extractor'], config['run']['cuda'])

    pose_cfg = config['test'].get('pose_cfg', None)
    position_variance = config['test'].get('position_variance', None)

    # ------------------
    # Initialise Dataset
    # ------------------
    src_dataset = CocoRaster(config['datasets']['test']['path'],
                             rasters=raster_names,
                             raster_paths=raster_paths,
                             live_raster=True,
                             raster_sizes=raster_sizes,
                             img_transform=transforms.ToTensor(),
                             raster_transforms=raster_transforms,
                             categories=True,
                             num_classes=config['test']['num_classes'],
                             pose_cfg=pose_cfg,
                             position_variance=position_variance,
                             image_ids=True,
                             )
    cache_dir = config['run'].get('cache_dir', None)
    if cache_dir:
        cache_test_dir = os.path.join(cache_dir, 'train')
    else:
        cache_test_dir = None
    dataset = LatentGenerator(dataset=src_dataset, encoders=extractors,
                              cache_dir=cache_test_dir)

    return dataset

def main(cfg):
    # -----------------
    # Check parameters
    # -----------------
    CUDA = cfg['run']['cuda']
    scratch_dir = cfg['run']['scratch_dir']
    os.makedirs(scratch_dir, exist_ok=True)
    verbose = cfg.get('verbose', True)
    if CUDA and cfg['run']['cuda_latent']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # TODO - hacky. This is to make pyro guide go on GPU.

    dataset = load_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    latent_model = load_latent_model(cfg['model'], cfg['run']['cuda_latent'])

    if CUDA and cfg['run']['cuda_latent']:
        try:
            latent_model.cuda()
        except AttributeError:
            pass


    batch_format = dataset.get_output_format()
    for n, bf in enumerate(batch_format):
        if bf == 'label':
            label_format_idx = n
        elif bf == 'image_id':
            iid_format_idx = n
    all_others = list(range(len(batch_format)))
    all_others.pop(label_format_idx)
    if label_format_idx < iid_format_idx:
        all_others.pop(iid_format_idx-1)
    else:
        all_others.pop(iid_format_idx)
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

    # ------------
    # Validation
    # ------------
    corrects = 0
    vcount = 0

    if verbose:
        pbar = tqdm(total=len(dataloader), desc="Evaluating")

    iid_to_pred = {}

    iid_to_plabel = {}

    conf = {x: np.zeros([cfg['test']['num_classes']]) for x in
            range(cfg['test']['num_classes'])}  # key=true_class, value=pred_class
    collected_ids = []
    for batch, data_batch in enumerate(dataloader):
        y_true = data_batch[label_format_idx]
        iid = int(data_batch[iid_format_idx])
        latent_list = [data_batch[idx].type(torch.float32) for idx in all_others]
        latent = torch.cat(latent_list, dim=1)

        # ytrue_label = torch.max(y_true, 1)[1].type(LongTensor)

        latent = latent.type(FloatTensor)

        y_pred = latent_model.predict(latent, cfg['test'].get('monte_carlo_samples', 32))
        y_pred = y_pred.squeeze()

        iid_to_pred[iid] = y_pred
        iid_to_plabel[iid] = int(y_pred.mean(axis=0).argmax(-1))

        y_pred_class = np.argmax(np.mean(y_pred, axis=0), axis=-1)

        y_true_class = np.argmax(y_true.detach().cpu().numpy().squeeze())

        if y_true_class == y_pred_class:
            corrects += 1

        vcount += 1

        # ---------------------
        # Confusion Matrix
        # ---------------------
        conf[y_true_class][y_pred_class] += 1

        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()

    results = {
        "correct": corrects,
        "count": vcount,
        "acc": float(corrects) / float(vcount)
    }
    json.dump(results, open(os.path.join(scratch_dir, 'results.json'), 'w'), indent=4)

    coco = COCO(cfg['datasets']['test']['path'])
    images = []
    anns = []
    collected_ids = list(iid_to_plabel.keys())
    for image in coco.loadImgs(coco.getImgIds()):
        iid = image['id']
        if iid not in collected_ids:
            warnings.warn("Missing IID {}".format(iid))
            continue
        images.append(image)
        ann = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))[0]
        pann = copy.deepcopy(ann)
        pann['category_id'] = iid_to_plabel[iid]
        anns.append(pann)
    outds = copy.deepcopy(coco.dataset)

    json.dump(outds, open(os.path.join(scratch_dir, 'predictions.json'), 'w'), indent=4)

    conf_df = pd.DataFrame.from_dict(conf, orient='index',
                                     columns=[str(x) for x in range(cfg['test']['num_classes'])])

    return results

if __name__ == "__main__":
    main(args_to_cfg(get_args()))