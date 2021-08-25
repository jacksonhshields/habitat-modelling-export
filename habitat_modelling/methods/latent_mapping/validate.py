#!/usr/bin/env python3

import os
import json
import yaml
import argparse
import torch
import numpy as np
from habitat_modelling.methods.latent_mapping.utils import create_datasets,  load_latent_model
from tqdm import tqdm
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
        "cache_dir": args.cache_dir,
        "verbose": args.verbose
    }
    return cfg

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

    dataset, val_dataset = create_datasets(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    coco = COCO()

    latent_model = load_latent_model(cfg['model'], cfg['run']['cuda_latent'])

    if CUDA and cfg['run']['cuda_latent']:
        try:
            latent_model.cuda()
        except AttributeError:
            pass
        CUDA = True
        torch.set_default_tensor_type(
            'torch.cuda.FloatTensor')  # TODO - hacky. This is to make pyro guide go on GPU

    batch_format = dataset.get_output_format()
    for n, bf in enumerate(batch_format):
        if bf == 'label':
            label_format_idx = n
    all_others = list(range(len(batch_format)))
    all_others.pop(label_format_idx)

    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

    # ------------
    # Validation
    # ------------
    corrects = 0
    vcount = 0
    all_probs = []
    trues = []

    if verbose:
        pbar = tqdm(total=len(dataloader), desc="Evaluating")

    for batch, data_batch in enumerate(dataloader):
        y_true = data_batch[label_format_idx]
        latent_list = [data_batch[idx].type(torch.float32) for idx in all_others]
        latent = torch.cat(latent_list, dim=1)

        ytrue_label = torch.max(y_true, 1)[1].type(LongTensor)

        latent = latent.type(FloatTensor)

        y_pred = latent_model.predict(latent, cfg['validation'].get('monte_carlo_samples', 32))
        y_pred = y_pred.squeeze()

        corrects += (y_pred.argmax(-1) == ytrue_label.detach().cpu().numpy()).sum().item()

        vcount += ytrue_label.size(0)

        if verbose:
            pbar.update(1)
    results = {
        "correct": corrects,
        "count": vcount,
        "acc": float(corrects) / float(vcount)
    }
    json.dump(results, open(os.path.join(scratch_dir, 'results.json')))

    if verbose:
        pbar.close()

        


if __name__ == "__main__":
    main(args_to_cfg(get_args()))