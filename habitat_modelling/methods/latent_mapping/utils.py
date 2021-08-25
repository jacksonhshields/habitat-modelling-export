import torch
import pyro
import gpytorch
import os
import json
import numpy as np
from torchvision import transforms
from habitat_modelling.datasets.torch.coco_habitat import CocoRaster
from habitat_modelling.utils.general_utils import get_dict_arguments
from .feature_extractors import BathymetryEncoder, LatentGenerator, RawFeatureGenerator
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
from habitat_modelling.methods.latent_mapping.models.latent_bnn import LatentBNN1,LatentBNN2
from habitat_modelling.methods.latent_mapping.models.latent_dense import DenseLatentModel, DropLatentModel2, DropLatentModel3, DropLatentNeighbours2, DropLatentNeighbours3
from habitat_modelling.methods.latent_mapping.models.latent_bayesnn import BayesNN3
from habitat_modelling.methods.latent_mapping.models.latent_gp import MulticlassGPClassificationModel, DKLModel, DKLDense, GPClassifierInference, GPSKWrapper



def to_float_tensor(x):
    if x.dtype == np.uint8:  # Normalise if an integer
        x = x / 255
    return torch.FloatTensor(x)

def to_float(x):
    if x.dtype == np.uint8:
        x = x/255
    return x

def load_feature_extractor(extractor_config, CUDA):
    if extractor_config['type'] == "bathymetry_encoder" or extractor_config['type'] == "bathymetry_autoencoder":
        if isinstance(extractor_config['params'], str):
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))
        else:
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))

        extractor = BathymetryEncoder(extractor_dictionary, extractor_config['weights'], CUDA)
    else:
        raise ValueError("Extractor type")
    return extractor



def create_datasets(config):
    raster_names = []
    raster_sizes = []
    raster_paths = []
    raster_transforms = {}
    extractors = {}

    # See if you want to use raw features
    use_raw_features = config.get('use_raw_features', False)

    if use_raw_features:  # Uses features directly from the raster
        for name, entry in config['rasters'].items():
            raster_names.append(name)
            raster_paths.append(entry['path'])
            raster_sizes.append(entry['size'])
            raster_transforms[name] = to_float_tensor
    else:  # uses an autoencoder for feature extraction
        if 'extractors' in config: # RECCOMENDED - this is compatible with geoinference
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

    pose_cfg = config['training'].get('pose_cfg', None)
    position_variance = config['training'].get('position_variance', None)

    # ------------------
    # Initialise Dataset
    # ------------------
    src_dataset = CocoRaster(config['datasets']['train']['path'],
                             rasters=raster_names,
                             raster_paths=raster_paths,
                             live_raster=True,
                             raster_sizes=raster_sizes,
                             img_transform=transforms.ToTensor(),
                             raster_transforms=raster_transforms,
                             categories=True,
                             num_classes=config['training']['num_classes'],
                             pose_cfg=pose_cfg,
                             position_variance=position_variance
                             )
    cache_dir = config['run'].get('cache_dir', None)
    if cache_dir:
        cache_train_dir = os.path.join(cache_dir, 'train')
        cache_val_dir = os.path.join(cache_dir, 'val')
    else:
        cache_train_dir = None
        cache_val_dir = None
    if use_raw_features:  # Uses features directly from the raster
        # raise NotImplementedError("TODO")
        dataset = RawFeatureGenerator(dataset=src_dataset, cache_dir=cache_train_dir)  # TODO cache
    else:  # uses an autoencoder for feature extraction
        dataset = LatentGenerator(dataset=src_dataset, encoders=extractors,
                                  cache_dir=cache_train_dir)

    if 'val' in config['datasets']:
        src_val_dataset = CocoRaster(config['datasets']['val']['path'],
                                     rasters=raster_names,
                                     raster_paths=raster_paths,
                                     live_raster=True,
                                     raster_sizes=raster_sizes,
                                     img_transform=transforms.ToTensor(),
                                     raster_transforms=raster_transforms,
                                     categories=True,
                                     num_classes=config['training']['num_classes'],
                                     pose_cfg=pose_cfg,
                                     position_variance=position_variance
                                     )
        if use_raw_features:  # Uses features directly from the raster
            val_dataset = RawFeatureGenerator(dataset=src_val_dataset, cache_dir=cache_val_dir)
        else:  # Uses an autoencoder for feature extraction
            val_dataset = LatentGenerator(dataset=src_val_dataset, encoders=extractors,
                                          cache_dir=cache_val_dir)
    else:
        val_dataset = None
    return dataset, val_dataset


def preprocess_datasets(dataset, val_dataset, preprocess_cfg):
    if preprocess_cfg['type'] == "standard":
        preprocess_registry = {}
        for k in dataset.cache_dict.keys():
            if 'latent' not in k:
                continue
            scaler = StandardScaler().fit(dataset.cache_dict[k])
            preprocess_registry[k] = scaler
            dataset.cache_dict[k] = scaler.transform(dataset.cache_dict[k])
            if val_dataset is not None:
                val_dataset.cache_dict[k] = scaler.transform(val_dataset.cache_dict[k])
    else:
        raise ValueError("Preprocess type not supported")

    if preprocess_cfg.get('save_pickle_dir', False):
        os.makedirs(preprocess_cfg['save_pickle_dir'],exist_ok=True)
        for k,v in preprocess_registry.items():
            pickle.dump(v, open(os.path.join(preprocess_cfg['save_pickle_dir'], k + '.pkl'), 'wb'))
    elif preprocess_cfg.get('save_pickle', False):
        pickle.dump(preprocess_registry, open(preprocess_cfg['save_pickle'], 'wb'))
    else:
        warnings.warn("Save pickle not given - this pickle will be needed for inference")
    return dataset, val_dataset



def load_latent_model(model_cfg, CUDA):
    # -------------------------------
    # Initialise latent mapping model
    # -------------------------------
    if isinstance(model_cfg['params'], dict):
        classifier_params = model_cfg['params']
    else:
        classifier_params = json.load(open(model_cfg['params'], 'r'))
    
    if model_cfg['type'] == "bnn1":
        # Initialise the model
        latent_model = LatentBNN1(**classifier_params)
        # If pyro params are given, load weights, params separately
        if 'pyro_params' in model_cfg:
            latent_model.load_state_dict(torch.load(model_cfg['weights']))
            pyro.get_param_store().load(model_cfg['pyro_params'])
        else:
            # If no pyro params are given, assumes latent model and params are stored together.
            # Load the model dict
            model_dict = torch.load(model_cfg['weights'])
            # Load the model state dict (weights)
            latent_model.load_state_dict(model_dict['state_dict'])
            # Load + set the params store
            pyro.get_param_store().set_state(model_dict['pyro_params'])

    elif model_cfg['type'] == "bnn2":
        # Initialise the model
        latent_model = LatentBNN2(**classifier_params)

        # If pyro params are given, load weights, params separately
        if 'pyro_params' in model_cfg:
            latent_model.load_state_dict(torch.load(model_cfg['weights']))
            pyro.get_param_store().load(model_cfg['pyro_params'])
        else:
            # If no pyro params are given, assumes latent model and params are stored together.
            # Load the model dict
            model_dict = torch.load(model_cfg['weights'])
            # Load the model state dict (weights)
            latent_model.load_state_dict(model_dict['state_dict'])
            # Load + set the params store
            pyro.get_param_store().set_state(model_dict['pyro_params'])

    elif model_cfg['type'] == "bayesnn3":
        latent_model = BayesNN3(**classifier_params)
        # If pyro params are given, load weights, params separately
        if 'pyro_params' in model_cfg and model_cfg['pyro_params'] is not None:
            latent_model.load_state_dict(torch.load(model_cfg['weights']))
            pyro.get_param_store().load(model_cfg['pyro_params'])
        else:
            # If no pyro params are given, assumes latent model and params are stored together.
            # Load the model dict
            model_dict = torch.load(**classifier_params)
            # Load the model state dict (weights)
            latent_model.load_state_dict(model_dict['state_dict'])
            # Load + set the params store
            pyro.get_param_store().set_state(model_dict['pyro_params'])

    elif model_cfg['type'] == "dense":
        # Initialise the model
        latent_model = DenseLatentModel(**classifier_params)
        # Load the state dict (weights)
        latent_model.load_state_dict(torch.load(model_cfg['weights']))
    elif model_cfg['type'] == "dropn2":
        latent_model = DropLatentNeighbours2(**classifier_params)
        latent_model.load_state_dict(torch.load(model_cfg['weights']))
    elif model_cfg['type'] == "dropn3":
        latent_model = DropLatentNeighbours3(**classifier_params)
        latent_model.load_state_dict(torch.load(model_cfg['weights']))
    elif model_cfg['type'] == "drop":
        if len(classifier_params['layer_neurons']) == 2:
            latent_model = DropLatentModel2(**classifier_params)
        elif len(classifier_params['layer_neurons']) == 3:
            latent_model = DropLatentModel3(**classifier_params)
        else:
            raise ValueError("Model depth not supported")
        # Load the weights
        latent_model.load_state_dict(torch.load(model_cfg['weights']))
    elif model_cfg['type'] == "gp_grid":
        gpc = MulticlassGPClassificationModel(**classifier_params)
        gpc.load_state_dict(torch.load(model_cfg['weights']))
        
        if isinstance(model_cfg['likelihood']['params'], dict):
            likelihood_params = model_cfg['likelihood']['params']
        else:
            likelihood_params = json.load(open(model_cfg['likelihood']['params'], 'r'))

        
        # likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=gpc.num_dim, **model_cfg['likelihood']['params'])  # model_cfg['likelihood']['params'] should define num_classes
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(**likelihood_params)  # model_cfg['likelihood']['params'] should define num_classes, num_features
        likelihood.load_state_dict(torch.load(model_cfg['likelihood']['weights']))
        if CUDA:
            gpc.cuda()
            likelihood.cuda()
        latent_model = GPClassifierInference(gpc, likelihood)
    elif model_cfg['type'] == "gp_dkl":
        classifier_params = json.load(**classifier_params)
        gpc = DKLDense(**classifier_params)
        gpc.load_state_dict(torch.load(model_cfg['weights']))
        
        if isinstance(model_cfg['likelihood']['params'], dict):
            likelihood_params = model_cfg['likelihood']['params']
        else:
            likelihood_params = json.load(open(model_cfg['likelihood']['params'], 'r'))
        
        # likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=gpc.layer_neurons[-1], **model_cfg['likelihood']['params'])  # model_cfg['likelihood']['params'] should define num_classes
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(**likelihood_params)  # model_cfg['likelihood']['params'] should define num_classes, num_features
        likelihood.load_state_dict(torch.load(model_cfg['likelihood']['weights']))
        if CUDA:
            gpc.cuda()
            likelihood.cuda()
        latent_model = GPClassifierInference(gpc, likelihood)

    elif "gpsk" in model_cfg['type']:
        classifier = pickle.load(**classifier_params)
        latent_model = GPSKWrapper(classifier)
    else:
        raise ValueError("Model type not selected")

    return latent_model
