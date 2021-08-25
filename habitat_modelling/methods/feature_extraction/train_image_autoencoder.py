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
from habitat_modelling.datasets.torch.general import CocoImage
from habitat_modelling.ml.torch.transforms.image_transforms import image_transforms_from_cfg
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
import neptune
import itertools
import csv
import ast
from autoencoders import ImageAutoencoder
from plotting import sample_iae
from habitat_modelling.utils.general_utils import get_dict_arguments

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

def main(cfg):
    CUDA = cfg['run']['cuda']
    scratch_dir = cfg['run']['scratch_dir']
    os.makedirs(scratch_dir, exist_ok=True)
    model_dir = os.path.join(scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(scratch_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    image_dir = os.path.join(scratch_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)


    if cfg['run']['neptune']:
        try:
            neptune.init('jacksonhshields/habitat-modelling')
            # neptune.init('jacksonhshields/sandbox')
            exp = neptune.create_experiment(cfg['run']['neptune'], params=cfg)
        except:
            warnings.warn("neptune couldn't be initiated")
            exp = None
            cfg['run']['neptune'] = None
    else:
        exp = None

    iae = ImageAutoencoder(
        image_shape=(cfg['model']['input_shape'][2], cfg['model']['input_shape'][0], cfg['model']['input_shape'][1]),  # Put channels first
        image_latent_dim=cfg['model']['latent_dim'],
        image_conv_filters=cfg['model']['filters'],
        image_block_type=cfg['model'].get('block_type', 'plain'),
        image_activation_cfg=None
    )
    iae.create_image_autoencoder()
    iae.dump_params_to_file(os.path.join(model_dir, 'model_params.json'))

    mse_loss = torch.nn.MSELoss()

    if CUDA:
        iae.image_encoder.cuda()
        iae.image_generator.cuda()
        mse_loss.cuda()

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    img_transforms, post_processor = image_transforms_from_cfg(cfg['training']['transforms'])

    dataset = CocoImage(cfg['datasets']['train'],
                        img_transform=img_transforms,
                        use_categories=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    pil_transform = transforms.ToPILImage()

    optimizer = torch.optim.Adam(itertools.chain(iae.image_encoder.parameters(), iae.image_generator.parameters()), lr=cfg['training']['lr'], betas=(0.5, 0.999))
    # ---------------------
    # LR scheduler
    # ---------------------
    if 'schedule' in cfg['training']:
        lr_schedule_type = cfg['training']['schedule']['type']
        if lr_schedule_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **get_dict_arguments(cfg['training']['schedule'], rm_keys=['type']))
        elif lr_schedule_type is not None:
            warnings.warn("LR Schedule %s not implemented" %lr_schedule_type)
            scheduler = None
        else:
            scheduler = None
    else:
        scheduler = None
    # Log results to a csv
    with open(os.path.join(log_dir, 'training_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Batch", "i2i Loss"])

    for epoch in range(cfg['training']['epochs']):
        for batch, (img_batch,) in enumerate(dataloader):

            batch_size = img_batch.shape[0]

            # ---------------------------
            # Train the autoencoder
            # ---------------------------

            optimizer.zero_grad()

            real_imgs = Variable(img_batch.type(Tensor))

            zi = iae.image_encoder(real_imgs)
            i2i = iae.image_generator(zi)

            i2i_loss = mse_loss(i2i, real_imgs)

            i2i_loss.backward()

            optimizer.step()

            print("[Epoch %d][Batch %d][i2i Loss %.2f]"
                %(epoch, batch, i2i_loss.item()))

            with open(os.path.join(log_dir, 'training_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch, i2i_loss.item()])

            if exp:
                exp.send_metric('i2i_loss', i2i_loss.item())

        sample_iae(
            real_imgs=img_batch,
            i2i=i2i.detach().cpu(),
            epoch=epoch,
            save_image_dir=image_dir,
            img_postprocessor=post_processor,
            num_imgs=batch_size,
            neptune_exp=exp
        )

        if epoch % cfg['training']['save_model_freq'] == 0:
            # Save state dict
            torch.save(iae.image_encoder.state_dict(), os.path.join(model_dir, 'iae_encoder_%d.pt' % epoch))
            torch.save(iae.image_generator.state_dict(), os.path.join(model_dir, 'iae_generator_%d.pt' % epoch))

        if scheduler:
            scheduler.step()

    if exp:
        neptune.stop()

if __name__ == "__main__":
    main(args_to_cfg(get_args()))
