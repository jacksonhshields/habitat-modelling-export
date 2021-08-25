#!/usr/bin/env python3
import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gpytorch
try:
 from osgeo import gdal
except ImportError:
 import gdal
import json
import yaml
import warnings
import csv
import argparse

import pandas as pd
import neptune

from habitat_modelling.datasets.torch.coco_habitat import CocoImgRaster
from habitat_modelling.datasets.utils import calculate_class_weights
from habitat_modelling.utils.display_utils import plot_confusion_matrix
from habitat_modelling.ml.torch.transforms.image_transforms import create_image_transform_list_from_args
from torchvision import transforms
from habitat_modelling.datasets.torch.coco_habitat import CocoImgRaster, CocoRaster

from habitat_modelling.methods.latent_mapping.feature_extractors import BathymetryEncoder, BathyLatentGenerator, CachedLatentGenerator, LatentGenerator
from habitat_modelling.methods.latent_mapping.utils import create_datasets, load_latent_model, preprocess_datasets
from habitat_modelling.utils.general_utils import get_dict_arguments

from habitat_modelling.methods.latent_mapping.models.latent_gp import MulticlassGPClassificationModel, DKLDense

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
    if 'preprocess' not in cfg or 'cache_dir' not in cfg['run']:
        raise ValueError("Preprocess needs to be given for GP methods. Also dataset needs to be cached")
    dataset, val_dataset = preprocess_datasets(dataset, val_dataset, cfg['preprocess'])

    # -------------------------------
    # Initialise latent mapping model
    # -------------------------------
    model_cfg = cfg['model']
    if model_cfg['type'] == "gp_grid":
        gpc = MulticlassGPClassificationModel(**model_cfg['params'])
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(**model_cfg['likelihood_params'])

    elif model_cfg['type'] == "gp_dkl":
        gpc = DKLDense(**model_cfg['params'])
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(**model_cfg['likelihood_params'])
    else:
        raise NotImplementedError("Model type not supported")
    gpc.dump_params(os.path.join(model_dir, 'model_params.json'))
    json.dump(model_cfg['likelihood_params'], open(os.path.join(model_dir, 'likelihood_params.json'), 'w'))

    warm_weights_path = model_cfg.get("weights", False)
    likelihood_path = model_cfg.get("likehihood_weights", False)
    if warm_weights_path and isinstance(warm_weights_path, str) and warm_weights_path != "":
        gpc.load_state_dict(torch.load(warm_weights_path))
    if likelihood_path and isinstance(likelihood_path, str) and likelihood_path != "":
        likelihood.load_state_dict(torch.load(likelihood_path))

    # ----------------
    # Initialise Logs
    # ----------------
    if cfg['run'].get('neptune', None):
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

    # Log results to a csv
    with open(os.path.join(log_dir, 'training_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Batch", "Loss"])

    # Log results to a csv
    with open(os.path.join(log_dir, 'validation_logs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epoch", "Val_Loss", "Val_Acc"])


    # -----------------------
    # Initialise data loader
    # -----------------------
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    else:
        val_dataloader = None

    # -----------------------
    # Calculate Class Weights
    # -----------------------
    if cfg['training'].get('class_weights', None):
        if cfg['training'].get('manual_class_weights', None):
            class_weights_arr = np.array(cfg['training']['manual_class_weights'])
        else:
            class_weights_arr = calculate_class_weights(json.load(open(cfg['datasets']['train'], 'r')))
        print("Using class weights ", class_weights_arr)
        class_weights = torch.tensor(class_weights_arr).type(torch.FloatTensor)

    else:
        class_weights = None
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
        gpc.cuda()
        likelihood.cuda()

    # ---------------------
    # Create Optimizer
    # ---------------------
    lr = cfg['training']['optimizer']['lr']
    if model_cfg['type'] == "gp_grid":
        optimizer = torch.optim.SGD(
            [{'params': gpc.hyperparameters(), 'lr': lr * 0.01}, {'params': gpc.variational_parameters()},
             {'params': likelihood.parameters()}], **get_dict_arguments(cfg['training']['optimizer'], rm_keys=['type']))
        mll = gpytorch.mlls.VariationalELBO(likelihood, gpc, num_data=len(dataset))

    elif model_cfg['type'] == "gp_dkl":
        optimizer = torch.optim.SGD([
                                   {'params': gpc.dkl.feature_extractor.parameters(), 'weight_decay': 1e-4},
                                   {'params': gpc.dkl.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                                   {'params': gpc.dkl.gp_layer.variational_parameters()},
                                   {'params': likelihood.parameters()},
                                   ], **get_dict_arguments(cfg['training']['optimizer'], rm_keys=['type']))
        mll = gpytorch.mlls.VariationalELBO(likelihood, gpc.dkl.gp_layer, num_data=len(dataset))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * cfg['training']['epochs'], 0.75 * cfg['training']['epochs']], gamma=0.1)

    if cfg['training']['optimizer'].get('weights', False):
        optimizer.load_state_dict(torch.load(cfg['training']['optimizer']['weights']))

    # ---------------------
    # LR scheduler
    # ---------------------
    if cfg['training'].get('schedule', None):
        if cfg['training']['schedule']['type'] == "step":
            schedule_params = cfg['training']['schedule']['params']
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **schedule_params)
        elif cfg['training']['schedule']['type'] == "multi":
            schedule_params = cfg['training']['schedule']['params']
            if 'gamma' not in schedule_params:
                schedule_params['gamma'] = 0.1
            if 'milestones' not in schedule_params:
                schedule_params['milestones'] = [0.5 * cfg['training']['epochs'], 0.75 * cfg['training']['epochs']]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **schedule_params)
        else:
            warnings.warn("LR Schedule %s not implemented" % cfg['training']['schedule'])
            scheduler = None
    else:
        scheduler = None


    # --------------------
    # Create Tensors
    # --------------------
    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor

    gpc.train()
    # --------------------
    # Train
    # --------------------

    batch_format = dataset.get_output_format()
    for n, bf in enumerate(batch_format):
        if bf == 'label':
            label_format_idx = n
    all_others = list(range(len(batch_format)))
    all_others.pop(label_format_idx)

    for epoch in range(cfg['training']['epochs']):
        running_corrects = 0
        running_count = 0


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

            optimizer.zero_grad()

            latent_t = Variable(latent.type(FloatTensor))
            ytrue_label = Variable(torch.max(y_true, 1)[1].type(LongTensor))

            latent_t = Variable(latent_t)

            output = gpc(latent)
            loss = -mll(output, ytrue_label)

            loss.backward()

            optimizer.step()

            #corrects = np.sum(yt_cats == yp_cats)

            #batch_acc = float(corrects)/float(batch_size)

            print("[Epoch %d][Batch %d][Loss %.2f]"
                %(epoch, batch, loss.item()))

            with open(os.path.join(log_dir, 'training_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch, loss.item()])

            if exp:
                exp.send_metric('loss', loss)
                #exp.send_metric('acc', batch_acc)

            if cfg['training'].get('train_steps') and batch >= cfg['training']['train_steps']:
                break

        if epoch % cfg['training'].get('save_model_freq', 1) == 0:
            torch.save(gpc.state_dict(), os.path.join(model_dir, 'gpc_%d.pt' % epoch))
            torch.save(likelihood.state_dict(), os.path.join(model_dir, 'likelihood_%d.pt' % epoch))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer_%d.pt' % epoch))
            model_dict = {
                'model': gpc.state_dict(),
                'likelihood': likelihood.state_dict()
            }
            torch.save(model_dict, os.path.join(model_dir, 'gpc_state_params_%d.pt' % epoch))

        # --------------------------
        # Validation
        # --------------------------
        if val_dataset is not None:
            val_corrects = 0
            val_loss_sum = 0
            val_total = 0

            # ---------------------------------------
            # Initialise confusion matrix dictionary
            # ---------------------------------------
            conf = {x: np.zeros([cfg['training']['num_classes']]) for x in
                    range(cfg['training']['num_classes'])}  # key=true_class, value=pred_class

            # ---------------------
            # Normal Evaluation
            # ---------------------
            val_corrects = 0
            val_loss_sum = 0
            val_total = 0

            for batch, data_batch in enumerate(val_dataloader):
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

                with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):

                    ytrue_label = Variable(torch.max(y_true, 1)[1].type(LongTensor))

                    latent = latent.type(FloatTensor)

                    output = likelihood(gpc(latent))
                    #val_loss = -mll(output, ytrue_label)
                    val_loss = 0
                    y_pred_class = output.probs.mean(0).argmax(-1)
                    val_corrects += y_pred_class.eq(ytrue_label.view_as(y_pred_class)).cpu().sum()
                    val_total += 1
                    #val_loss_sum += val_loss.item()
                    val_loss_sum = 0
                #y_pred = y_pred.detach().cpu().numpy()
                #y_pred_class = np.argmax(y_pred, axis=1)
                #y_true_class = np.argmax(y_true.detach().cpu().numpy(), axis=1)
                #val_corrects += np.sum(y_pred_class == y_true_class)
                #val_total += 1

                # ---------------------
                # Confusion Matrix
                # ---------------------
                #for n in range(y_true_class.shape[0]):
                #    conf[y_true_class[n]][y_pred_class[n]] += 1

                if cfg['training'].get('val_steps') and batch >= cfg['training']['val_steps']:
                    break


            val_acc = float(val_corrects)/float(val_total)
            val_loss = val_loss_sum/float(val_total)
            if np.isnan(val_loss):
                val_loss = 0.0
            if exp:
                exp.send_metric('val_acc', val_acc)
                exp.send_metric('val_loss', val_loss)
            print("Validation Accuracy: ", val_acc)
            print("Validation Loss: ", val_loss)

            with open(os.path.join(log_dir, 'validation_logs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, val_loss, val_acc])


            #conf_df = pd.DataFrame.from_dict(conf, orient='index', columns=[str(x) for x in range(cfg['training']['num_classes'])])

            # print(conf_df)

            #plot_confusion_matrix(conf_df.to_numpy().astype(np.int32), conf_df.columns.values, save_path=os.path.join(image_dir, 'confusion_matrix_e%d.png'%epoch))

        if scheduler:
            scheduler.step()

    if exp:
        neptune.stop()

if __name__ == "__main__":
    main(args_to_cfg(get_args()))
