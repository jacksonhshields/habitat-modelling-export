#!/usr/bin/env python3
import os
import numpy as np
import sys
import json
import yaml
import torch

import torch.nn.functional as F
from torch.autograd import Variable

try:
 from osgeo import gdal
except ImportError:
 import gdal

import argparse

import warnings
import csv

# Pyro
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from torchvision import transforms
import pyro.optim
from habitat_modelling.datasets.torch.coco_habitat import CocoImgRaster, CocoRaster
from habitat_modelling.datasets.torch.general import ArrayDataset
from habitat_modelling.utils.display_utils import plot_confusion_matrix
from habitat_modelling.methods.latent_mapping.utils import create_datasets
# from .utils import create_datasets
import pandas as pd
# from .models.latent_bayesnn import BayesNN3
from habitat_modelling.methods.latent_mapping.models.latent_bayesnn import BayesNN3, BayesNN3Lift
import neptune
from habitat_modelling.utils.general_utils import get_dict_arguments



def to_float_tensor(x):
    if x.dtype == np.uint8:  # Normalise if an integer
        x = x / 255
    return torch.FloatTensor(x)

def to_float(x):
    if x.dtype == np.uint8:
        x = x/255
    return x

def no_op(x):
    return x


def get_args():
    parser = argparse.ArgumentParser(description="Trains a multimodal (image + bathy) autoencoder")
    parser.add_argument('--config', help="The configuration file", required=True)
    parser.add_argument('--scratch-dir', type=str, default="scratch/",
                        help="Directory to hold all scratch models, logs etc.")
    parser.add_argument('--cache-dir', type=str, help="The cache directory")
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
        "cache_dir": args.cache_dir
    }
    return cfg

def main(cfg):

    # -----------------
    # Check parameters
    # -----------------
    scratch_dir = cfg['run']['scratch_dir']
    os.makedirs(scratch_dir, exist_ok=True)

    # --------------------
    # Create Tensors
    # --------------------
    CUDA = cfg['run']['cuda']
    if CUDA:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

    # -----------------------
    # Create Log Directories
    # -----------------------
    log_dir = os.path.join(scratch_dir, 'logs')
    model_dir = os.path.join(scratch_dir, 'models')
    image_dir = os.path.join(scratch_dir, 'images')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    os.makedirs(os.path.join(log_dir, 'detailed_validation'), exist_ok=True)

    dataset, val_dataset = create_datasets(cfg)

    # -------------------------------
    # Initialise latent mapping model
    # -------------------------------
    if cfg['model']['type'] == "bayesnn3":
        bayesnn = BayesNN3(**cfg['model']['params'])
    elif cfg['model']['type'] == "bayesnn3lift":
        bayesnn = BayesNN3Lift(**cfg['model']['params'])  # TODO
    bayesnn.dump_params(os.path.join(model_dir, 'model_params.json'))

    if cfg['model'].get('warm_weights', None) and cfg['model'].get('warm_pyro_params', None):
        bayesnn.load_state_dict(torch.load(cfg['model']['warm_weights']))
        pyro.get_param_store().load(cfg['model']['warm_pyro_params'])
    elif cfg['model'].get('warm_weights', None):
        bayesnn.load_state_dict(torch.load(cfg['model']['warm_weights']))
        warnings.warn("Need to load pyro params as well. Warm starting probably didn't work.")
    elif cfg['model'].get('warm_pyro_params', None):
        pyro.get_param_store().load(cfg['model']['warm_pyro_params'])
        warnings.warn("Need to load pyro params as well. Warm starting probably didn't work.")

    # ----------------
    # Initialise Logs
    # ----------------
    if cfg['run'].get('neptune', None):
        try:
            neptune.init('jacksonhshields/habitat-modelling')
            exp = neptune.create_experiment(cfg['run']['neptune'], params=cfg)
        except:
            warnings.warn("neptune couldn't be initiated")
            exp = None
            cfg['run']['neptune'] = None
    else:
        exp = None

    # Log results to a csv
    with open(os.path.join(log_dir, 'training_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Batch", "Loss"])

    # Log results to a csv
    with open(os.path.join(log_dir, 'validation_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Val_Acc", "Sel_Val_Acc"])

    # Log results to a csv
    class_names = ["Class_%d" % x for x in range(cfg['training']['num_classes'])]
    with open(os.path.join(log_dir, 'validation_per_class.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Val_Acc"] + class_names)


    # -----------------------
    # Initialise data loader
    # -----------------------
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    else:
        val_dataloader = None
    # -----------------------
    # Loss
    # -----------------------
    cat_loss = torch.nn.CrossEntropyLoss()
    cat_loss = torch.nn.BCELoss()

    # ----------------------
    # CUDA
    # ----------------------
    if CUDA:
        cat_loss.cuda()
        bayesnn.cuda()

    # ---------------------
    # Create Optimizer
    # ---------------------
    # optimizer = torch.optim.Adam


    # Adam optimizer
    if cfg['training']['optimizer']['type']:
        optim_args = get_dict_arguments(cfg['training']['optimizer'], rm_keys=['type','warm_optimizer'])
        optimizer = pyro.optim.Adam(optim_args)
        # optimizer = torch.optim.Adam

    if cfg['training']['optimizer'].get('warm_optimizer', None):
        optimizer.load_state_dict(torch.load(cfg['training']['optimizer']['warm_optimizer']))

    # optim_args = {'lr': args.lr, 'momentum': 0.9, 'nesterov': True}
    # optim = pyro.optim.SGD(optim_args)

    if cfg['training'].get('lr_schedule', None) is not None:
        optimizer = torch.optim.Adam  # TODO support others
        lr_scheduler_args = get_dict_arguments(cfg['training']['lr_schedule'], rm_keys=['type'])
        if cfg['training']['lr_schedule']['type'] == "step":
            scheduler = pyro.optim.StepLR(
                {'optimizer': optimizer,
                 'optim_args': optim_args,
                 'gamma': lr_scheduler_args['gamma'],
                 'step_size': lr_scheduler_args['step_size']})
            # scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': adam_args, 'gamma': 0.996})
        else:
            raise ValueError("LR scheduler %s not valid" % cfg['training']['step'])
    else:
        scheduler = None


    elbo = TraceMeanField_ELBO()
    if scheduler is not None:
        svi = SVI(bayesnn.model, bayesnn.guide, scheduler, elbo)
    else:
        svi = SVI(bayesnn.model, bayesnn.guide, optimizer, elbo)

    # --------------------
    # Create Tensors
    # --------------------
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor


    if cfg['training'].get('class_weights', None):
        use_class_weights = cfg['training']['class_weights']['enabled']
        class_weights = cfg['training']['class_weights'].get('weights', None)
        if class_weights is None:  # Calculate class weights
            counts = dataset.get_category_counts()
            class_weights = np.zeros(counts.shape)
            for n in range(len(counts)):  # This is sklearn 'balanced' class weights
                if counts[n] == 0:
                    class_weights[n] = 1.0
                else:
                    class_weights[n] = np.sum(counts) / (len(counts) * counts[n])

    else:
        use_class_weights = False
        class_weights = None

    # --------------------
    # Train
    # --------------------

    batch_format = dataset.get_output_format()
    for n, bf in enumerate(batch_format):
        if bf == 'label':
            label_format_idx = n
    all_others = list(range(len(batch_format)))
    all_others.pop(label_format_idx)

    loss = 0
    for epoch in range(cfg['training']['epochs']):
        running_corrects = 0
        running_count = 0
        running_loss = 0

        kl_factor = dataloader.batch_size / len(dataloader.dataset)


        for batch, data_batch in enumerate(dataloader):

            y_true = data_batch[label_format_idx]
            latent_list = [data_batch[idx].type(torch.float32) for idx in all_others]
            latent = torch.cat(latent_list, dim=1)

            # Test for invalid
            depths = latent.detach().cpu().numpy()[:, -1]
            if np.any(depths == 0.0):
                continue

            if torch.any(torch.isnan(latent)):
                continue

            # Batch size is not guaranteed with dataloader
            batch_size = latent.size(0)

            ytrue_label = Variable(torch.max(y_true, 1)[1].type(LongTensor))

            latent = Variable(latent.type(FloatTensor))

            # model_output = bayesnn.model(latent, ytrue_label)
            # guide_output = bayesnn.guide(latent)
            if use_class_weights:
                class_scale = class_weights[int(ytrue_label.detach().cpu().numpy())]
            else:
                class_scale = 1.0

            batch_loss = svi.step(latent, ytrue_label, kl_factor=kl_factor, scale=class_scale)

            pred = bayesnn(latent, n_samples=1).mean(0)

            running_corrects += (pred.argmax(-1) == ytrue_label).sum().item()

            running_count += ytrue_label.size(0)

            running_loss += batch_loss

            print("[Epoch %d][Batch %d][Loss %.2f]"
                %(epoch, batch, batch_loss))

            with open(os.path.join(log_dir, 'training_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch, batch_loss])

            if exp:
                exp.send_metric('loss', batch_loss)

            if cfg['training'].get('train_steps',None) and batch >= cfg['training']['train_steps']:
                break

        if epoch % cfg['training'].get('save_model_freq',1) == 0:
            # Save state dict
            # bnn.save_state(os.path.join(model_dir, 'bnn_%d.pt' % epoch))
            torch.save(bayesnn.state_dict(), os.path.join(model_dir, 'bnn_%d.pt' % epoch))
            if scheduler is None:
                optimizer.save(os.path.join(model_dir, 'optimizer_%d.pt' % epoch))
            pyro.get_param_store().save(os.path.join(model_dir, 'pyro_params_%d.pt' % epoch))
            # scheduler.save(os.path.join(model_dir, 'optimizer_%d.pt'))
            model_dict = {
                'state_dict': bayesnn.state_dict(),
                'pyro_params': pyro.get_param_store().get_state()
            }
            torch.save(model_dict, os.path.join(model_dir, 'bnn_state_params_%d.pt' % epoch))



        # ------------
        # Validation
        # ------------

        if val_dataset is not None:
            corrects = 0
            vcount = 0

            # Selective correct/counts
            sel_correct = 0
            sel_count = 0

            # ---------------------------------------
            # Initialise confusion matrix dictionary
            # ---------------------------------------
            conf = {x: np.zeros([cfg['training']['num_classes']]) for x in range(cfg['training']['num_classes'])}  # key=true_class, value=pred_class

            for batch, data_batch in enumerate(val_dataloader):

                y_true = data_batch[label_format_idx]
                latent_list = [data_batch[idx].type(torch.float32) for idx in all_others]
                latent = torch.cat(latent_list, dim=1)

                # Test for invalid
                depths = latent.detach().cpu().numpy()[:,-1]
                if np.any(depths == 0.0):
                    continue

                if torch.any(torch.isnan(latent)):
                    continue

                with torch.no_grad():
                    batch_size = latent.size(0)

                    latent = Variable(latent.type(FloatTensor))

                    y_pred = bayesnn.predict(latent, num_samples=cfg['training'].get('inference_samples',16))
                    y_pred = np.squeeze(y_pred, axis=1)
                    y_pred_class = np.argmax(np.mean(y_pred, axis=0), axis=-1)

                    y_true_class = np.argmax(y_true.detach().cpu().numpy().squeeze())

                    # ---------------------
                    # Confusion Matrix
                    # ---------------------
                    conf[y_true_class][y_pred_class] += 1


                correct = np.sum(y_pred_class == np.argmax(y_true.detach().cpu().numpy(), axis=1), axis=0)
                corrects += correct
                vcount += batch_size

                if cfg['training'].get('val_steps', None) and batch >= cfg['training']['val_steps']:
                    break

            val_acc = float(corrects) / float(vcount)
            print("Val Acc %.3f" %val_acc)
            if exp:
                exp.send_metric('val_acc', val_acc)

            conf_df = pd.DataFrame.from_dict(conf, orient='index', columns=[str(x) for x in range(cfg['training']['num_classes'])])

            # plot_confusion_matrix(conf_df.to_numpy().astype(np.int32), conf_df.columns.values, save_path=os.path.join(image_dir, 'confusion_matrix_e%d.png'%epoch))

            with open(os.path.join(log_dir, 'validation_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, val_acc])

            per_class_accuracies = [conf[x][x]/np.sum(conf[x]) for x in conf.keys()]
            with open(os.path.join(log_dir, 'validation_per_class.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, val_acc] + per_class_accuracies)
    if scheduler is not None:
        scheduler.step()

    if exp:
        neptune.stop()

if __name__ == "__main__":
    main(args_to_cfg(get_args()))
