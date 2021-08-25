#!/usr/bin/env python3
import os
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from habitat_modelling.datasets.torch.coco_habitat import CocoRaster
from habitat_modelling.datasets.torch.sample_habitat import SampleRaster
from habitat_modelling.utils.raster_utils import raster_lists_from_cfg
from habitat_modelling.utils.general_utils import get_dict_arguments
from plotting import sample_bae, sample_back_ae, sample_bathy_back_ae
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
import neptune
import itertools
import csv
import ast
import shapely
import shapely.geometry
from autoencoders import BathyAutoencoder, BathyVariationalAutoencoder
from plotting import sample_bae, sample_bathy_back_ae, sample_back_ae
from torch.utils.data import RandomSampler


def get_args():
    parser = argparse.ArgumentParser(description="Trains a multimodal (image + bathy) autoencoder")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default="scratch/",
                        help="Directory to hold all scratch models, logs etc.")
    parser.add_argument('--cuda', action="store_true", help="Whether to use cuda")
    parser.add_argument('--neptune', type=str,
                        help="The neptune experiment name. Omitting this means neptune isn't used.")

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
    }
    return cfg

class KLSchedule:
    def __init__(self, epochs, alphas):
        self.epochs = epochs
        self.alphas = alphas
        self.idx = 0
    def get_kl_alpha(self, epoch):
        if epoch > self.epochs[self.idx]:
            self.idx += 1
        if self.idx >= len(self.epochs):
            self.idx = len(self.epochs) - 1
        return self.alphas[self.idx]




def main(cfg):
    CUDA = cfg['run']['cuda']
    os.makedirs(cfg['run']['scratch_dir'], exist_ok=True)
    model_dir = os.path.join(cfg['run']['scratch_dir'], 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(cfg['run']['scratch_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    image_dir = os.path.join(cfg['run']['scratch_dir'], 'images')
    os.makedirs(image_dir, exist_ok=True)
    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    if cfg.get('neptune', None):
        try:
            neptune.init('jacksonhshields/habitat-modelling')
            # neptune.init('jacksonhshields/sandbox')
            exp = neptune.create_experiment(cfg['neptune'], params=cfg)
        except:
            warnings.warn("neptune couldn't be initiated")
            exp = None
            cfg['neptune'] = None
    else:
        exp = None

    if cfg['model'].get("type", "autoencoder") == "vae":
        bae = BathyVariationalAutoencoder(
            bathy_shape=(cfg['model']['input_shape'][2], cfg['model']['input_shape'][0], cfg['model']['input_shape'][1]),  # Put channels first
            bathy_latent_dim=cfg['model']['latent_dim'],
            bathy_conv_filters=cfg['model']['filters'],
            bathy_neurons=cfg['model']['neurons'],
            bathy_block_type=cfg['model']['block_type'],
            bathy_activation_cfg=None,
            tensor=Tensor
        )
    else:
        bae = BathyAutoencoder(
            bathy_shape=(
            cfg['model']['input_shape'][2], cfg['model']['input_shape'][0], cfg['model']['input_shape'][1]),
            # Put channels first
            bathy_latent_dim=cfg['model']['latent_dim'],
            bathy_conv_filters=cfg['model']['filters'],
            bathy_neurons=cfg['model']['neurons'],
            bathy_block_type=cfg['model']['block_type'],
            bathy_activation_cfg=None
        )

    bae.create_bathy_autoencoder()
    bae.dump_params_to_file(os.path.join(model_dir, 'model_params.json'))
    if cfg['model'].get('warm_encoder_weights', None):
        bae.bathy_encoder.load_state_dict(torch.load(cfg['model']['warm_encoder_weights']))
    if cfg['model'].get('warm_generator_weights', None):
        bae.bathy_generator.load_state_dict(torch.load(cfg['model']['warm_generator_weights']))

    mse_loss = torch.nn.MSELoss()

    if CUDA:
        bae.bathy_encoder.cuda()
        bae.bathy_generator.cuda()
        mse_loss.cuda()


    if 'dataset' in cfg and cfg.get('dataset', 'coco_path'):
        rasters, raster_paths, raster_sizes, raster_boundaries, raster_transforms = raster_lists_from_cfg(cfg['rasters'], lib='torch')
        dataset = CocoRaster(cfg['dataset']['coco_path'],
                             rasters=rasters,
                             live_raster=True,
                             raster_sizes=raster_sizes,
                             raster_transforms=raster_transforms,
                             )
    else:
        rasters, raster_paths, raster_sizes, raster_boundaries, raster_transforms = raster_lists_from_cfg(
            cfg['rasters'], lib='torch')

        dataset = SampleRaster(rasters=rasters,
                               raster_paths=raster_paths,
                               raster_sizes=raster_sizes,
                               raster_boundaries=raster_boundaries,
                               raster_transforms=raster_transforms,
                               length=cfg['training']['samples_per_epoch'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    if cfg['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(itertools.chain(bae.bathy_encoder.parameters(), bae.bathy_generator.parameters()),
                                     lr=cfg['training']['lr'], betas=(0.5, 0.999))
    else:
        raise ValueError("Optimizer %s not supported" % cfg['training']['optimizer'])

    # ---------------------
    # LR scheduler
    # ---------------------
    if 'schedule' in cfg['training']:
        if cfg['training']['schedule']['type'] == 'step':
            lr_schedule_params = get_dict_arguments(cfg['training']['schedule'], rm_keys=['type'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_schedule_params)
        else:
            raise NotImplementedError("Schedule %s not supported" % cfg['training']['schedule']['type'])
    else:
        scheduler = None

    # ---------------------
    # KL scheduler
    # ---------------------
    if 'kl_schedule' in cfg['training']:
        kl_schedule = KLSchedule(epochs=cfg['training']['kl_schedule']['epochs'], alphas=cfg['training']['kl_schedule']['alphas'])
    else:
        kl_schedule = KLSchedule(epochs=[0], alphas=[1.0])


    # Log results to a csv
    with open(os.path.join(log_dir, 'training_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Batch", "b2b Loss"])

    # HERE
    batch_format = dataset.get_output_format()
    raster_batch_idx = {}
    for r in rasters:
        if r == "bathymetry":
            raster_batch_idx["dmap"] = batch_format.index("dmap")
            raster_batch_idx["depth"] = batch_format.index("depth")
        else:
            raster_batch_idx[r] = batch_format.index(r)
    past_loss = -np.inf
    for epoch in range(cfg['training']['epochs']):
        loss_history = []
        kl_alpha = kl_schedule.get_kl_alpha(epoch)
        print("KL Alpha", kl_alpha)
        for batch, (patch_batch) in enumerate(dataloader):
            # ---------------------------
            # Train the autoencoder
            # ---------------------------

            optimizer.zero_grad()

            input_list = []
            real_depth = None

            data_ok = True

            for r in rasters:
                if r == "bathymetry":
                    real_bathy = Variable(patch_batch[raster_batch_idx["dmap"]]).type(Tensor)
                    real_depth = Variable(patch_batch[1].type(Tensor))
                    for n in range(real_bathy.size(0)):
                        if real_depth[n].item() == -9999.0:  # No data value
                            data_ok = False
                        if real_bathy[n].detach().cpu().numpy().max() == 0.0 and real_bathy[
                            n].detach().cpu().numpy().max() == 0.0:
                            data_ok = False
                    input_list.append(real_bathy)
                else:
                    real_patch = Variable(patch_batch[raster_batch_idx[r]]).type(Tensor)
                    input_list.append(real_patch)

            input_patch = torch.cat(input_list, dim=1)
            batch_size = input_patch.size(0)

            if not data_ok:
                print("Skipping due to bad data")
                continue

            if bae.bathy_encoder.variational:
                zb, mu, logvar = bae.bathy_encoder(input_patch)
                b2b = bae.bathy_generator(zb)
                b2b_loss = bae.loss_function(b2b, input_patch, mu, logvar, kl_alpha=kl_alpha)   # VAE loss
            else:
                zb = bae.bathy_encoder(input_patch)
                b2b = bae.bathy_generator(zb)
                b2b_loss = mse_loss(b2b, input_patch)

            if b2b_loss.item() > past_loss*2:
                print('here')
            past_loss = b2b_loss.item()

            loss_history.append(b2b_loss.item())
            if cfg['training']['focus_training']:
                cutoff_val = np.percentile(loss_history, cfg['training']['focus_percentile'])
                if b2b_loss.item() > cutoff_val:
                    b2b_loss.backward()
                    optimizer.step()
            else:
                b2b_loss.backward()
                optimizer.step()

            print("[Epoch %d][Batch %d][b2b Loss %.2f]"
                  % (epoch, batch, b2b_loss.item()))

            with open(os.path.join(log_dir, 'training_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch, b2b_loss.item()])

            if exp:
                exp.send_metric('b2b_loss', b2b_loss.item())

        if 'bathymetry' in rasters and 'backscatter' in rasters:
            sample_bathy_back_ae(
                input_patch=input_patch.detach().cpu().numpy(),
                b2b=b2b.detach().cpu().numpy(),
                epoch=epoch,
                save_image_dir=image_dir,
                num_imgs=batch_size,
                neptune_exp=exp
            )
        elif 'bathymetry' in rasters:
            sample_bae(
                real_bathy=input_patch.detach().cpu().numpy(),
                depth_batch=real_depth.detach().cpu().numpy(),
                b2b=b2b.detach().cpu().numpy(),
                epoch=epoch,
                save_image_dir=image_dir,
                num_imgs=batch_size,
                neptune_exp=exp
            )
        elif 'backscatter' in rasters:
            sample_back_ae(
                real_bathy=input_patch.detach().cpu().numpy(),
                b2b=b2b.detach().cpu().numpy(),
                epoch=epoch,
                save_image_dir=image_dir,
                num_imgs=batch_size,
                neptune_exp=exp
            )
        else:
            warnings.warn("Plotting only supported for bathymetry and backscatter")  # TODO fix this

        if epoch % cfg['training']['save_model_freq'] == 0:
            # Save state dict
            torch.save(bae.bathy_encoder.state_dict(),
                       os.path.join(model_dir, 'encoder_%d.pt' % (epoch)))
            torch.save(bae.bathy_generator.state_dict(),
                       os.path.join(model_dir, 'generator_%d.pt' % (epoch)))
        if scheduler:
            scheduler.step()
    if exp:
        neptune.stop()


if __name__ == "__main__":
    main(args_to_cfg(get_args()))
