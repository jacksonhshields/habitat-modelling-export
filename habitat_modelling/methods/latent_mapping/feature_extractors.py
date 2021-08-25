#!/usr/bin/env python3

import json
import numpy as np
import os
import copy
import cv2
import torch.nn as nn
import sys
import torch.utils.data as data
import warnings
from PIL import Image
import pickle
import torch
import torchvision
from torchvision.transforms import ToPILImage, ToTensor
from torch.autograd import Variable
from habitat_modelling.utils.raster_utils import retrieve_pixel_coords, extract_raster_patch, extract_geotransform, get_ranges
from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros
from habitat_modelling.utils.display_utils import colour_bathy_patch

from habitat_modelling.datasets.torch.coco_habitat import CocoImgRaster, CocoRaster
from habitat_modelling.methods.feature_extraction.autoencoders import BathyEncoder

class BathymetryEncoder:
    """
    Uses the bathymetry encoder to create a useful latent space.
    """
    def __init__(self, params, weights_path,  pre_transforms=None, cuda=True):
        self.params = params
        self.encoder = self.init_encoder_from_params(params)
        # Load the weights for the generator
        self.encoder.load_state_dict(torch.load(weights_path))
        self.encoder.eval()
        if cuda:
            self.encoder.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.pre_transforms = pre_transforms


    def init_encoder_from_params(self, params):
        """
        Initialises the info GAN model using the saved parameters. These don't match the input args so have to be remapped.
        Args:
            params:

        Returns:

        """
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '~/src/habitat-modelling/experiments'))


        activation = nn.LeakyReLU()
        params['activation'] = activation

        encoder = BathyEncoder(**params)
        return encoder

    def predict(self, dmap_batch):
        encoded = self.encoder(dmap_batch)
        return encoded

class BathymetryVariationalEncoder:
    """
    Uses the bathymetry encoder to create a useful latent space.
    """
    def __init__(self, params, weights_path,  pre_transforms=None, cuda=True):
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.params = params
        self.params['tensor'] = self.FloatTensor
        self.encoder = self.init_encoder_from_params(params)
        # Load the weights for the generator
        self.encoder.load_state_dict(torch.load(weights_path))
        self.encoder.eval()
        if cuda:
            self.encoder.cuda()
        self.pre_transforms = pre_transforms


    def init_encoder_from_params(self, params):
        """
        Initialises the info GAN model using the saved parameters. These don't match the input args so have to be remapped.
        Args:
            params:

        Returns:

        """
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '~/src/habitat-modelling/experiments'))


        activation = nn.LeakyReLU()
        params['activation'] = activation

        encoder = BathyEncoder(**params)
        return encoder

    def predict(self, dmap_batch):
        zb, mu, logvar = self.encoder(dmap_batch)
        return zb, mu, logvar

class BathyLatentGenerator(data.Dataset):
    def __init__(self, bathymetry_encoder, dataset, cuda=True):
        self.encoder = bathymetry_encoder
        self.dataset = dataset

        if cuda:
            self.encoder.encoder.cuda()
        self.cuda = cuda
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def __getitem__(self, idx):
        if isinstance(self.dataset, CocoImgRaster):
            _, dmap, depth, y_true = self.dataset[idx]
        elif isinstance(self.dataset, CocoRaster):
            dmap, depth, y_true = self.dataset[idx]
        else:
            raise ValueError("BathyLatentGenerator only supports CocoImgRaster and CocoRaster data generators")

        dmap_batch = dmap.unsqueeze(0)

        y_true = torch.tensor(y_true)
        depth = torch.tensor(depth)
        depth = depth.unsqueeze(0)
        with torch.no_grad():
            dmap_t = Variable(dmap_batch.type(self.FloatTensor))
            encoded = self.encoder.predict(dmap_t)
            latent = torch.cat((encoded.detach().cpu().squeeze(0), depth), dim=-1)
        return latent, y_true

    def __len__(self):
        return len(self.dataset)


class CachedLatentGenerator(data.Dataset):
    def __init__(self, latent_array, cat_array):
        self.latent_array = latent_array
        self.cat_array = cat_array

        # Calculate latent array
        self.dataset_length = self.latent_array.shape[0]
        if cat_array.shape[0] != self.dataset_length:
            raise ValueError("Length of Category Array is not the same as latent array")

    def __getitem__(self, idx):
        return self.latent_array[idx], self.cat_array[idx]

    def __len__(self):
        return self.dataset_length


class BathyLatentGeneratorNeighbours(data.Dataset):
    def __init__(self, bathymetry_encoder, dataset, cuda=True):
        self.encoder = bathymetry_encoder
        self.dataset = dataset

        if cuda:
            self.encoder.encoder.cuda()
        self.cuda = cuda
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def __getitem__(self, idx):
        if isinstance(self.dataset, CocoImgRaster):
            _, dmap, depth, y_true, neighbours = self.dataset[idx]
        elif isinstance(self.dataset, CocoRaster):
            dmap, depth, y_true, neighbours = self.dataset[idx]
        else:
            raise ValueError("BathyLatentGenerator only supports CocoImgRaster and CocoRaster data generators")

        dmap_batch = dmap.unsqueeze(0)
        neighbours = neighbours.type(torch.FloatTensor)
        neighbours = neighbours/torch.sum(neighbours)

        y_true = torch.tensor(y_true)
        depth = torch.tensor(depth)
        depth = depth.unsqueeze(0)
        with torch.no_grad():
            dmap_t = Variable(dmap_batch.type(self.FloatTensor))
            encoded = self.encoder.predict(dmap_t)
            latent = torch.cat((encoded.detach().cpu().squeeze(0), depth), dim=-1)
        return latent, y_true, neighbours

    def __len__(self):
        return len(self.dataset)


class RawFeatureGenerator(data.Dataset):
    """
    Raw Feature Generator.

    Inputs a cocoRaster dataset and concatenates the features.
    """
    def __init__(self, dataset, cache_dir=None, cuda=True):
        self.dataset = dataset
        self.rasters = self.dataset.rasters
        self.use_categories = self.dataset.use_categories
        self.use_neighbours = self.dataset.use_neighbours
        self.use_pose = self.dataset.use_pose
        self.use_image_ids = self.dataset.use_image_ids
        self.cuda = cuda
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if isinstance(self.dataset, CocoImgRaster):
            self.use_images = True
        elif isinstance(self.dataset, CocoRaster):
            self.use_images = False
        else:
            raise ValueError("RawFeatureGenerator only supports CocoImgRaster and CocoRaster data generators")

        self.cache_dict = {}
        self.cache_dir = cache_dir
        self.cache_ok = False

        if self.cache_dir and self.load_cached():
            self.cache_ok = True
            # self.clear_encoders()
        elif self.cache_dir and not self.load_cached():
            self.cache()
            if self.load_cached():
                self.cache_ok = True
                # self.clear_encoders()
            else:
                warnings.warn("Cache not working, using live instead")

    def get_category_counts(self):
        return self.dataset.get_category_counts()

    def __len__(self):
        if self.cache_ok:  # TODO hack remove
            output_list = self.get_output_format()
            return self.cache_dict[output_list[0]].shape[0]
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets data

        Args:
            idx:

        Returns:

        """
        if self.cache_ok:
            return self.get_cached(idx)
        else:
            return self.get_live(idx)

    def get_live(self, idx):
        """
        Gets live data
        Args:
            idx:

        Returns:

        """
        data_batch = self.dataset[idx]

        return_list = []

        data_idx = 0
        if self.use_images:
            if "image" in self.encoders.keys():
                image_latent = self.encoders["image"].predict(data_batch[data_idx])

                return_list.append(image_latent)
            data_idx += 1

        for r in self.rasters:
            # Process feature
            feat = data_batch[data_idx]  # feat comes out as 1,1
            feat = feat.squeeze(1)
            return_list.append(feat.squeeze(1))
            data_idx += 1

        if self.use_categories:
            return_list.append(data_batch[data_idx])
            data_idx += 1

        if self.use_neighbours:
            neighbours = data_batch[data_idx]
            neighbours = neighbours.type(torch.FloatTensor)
            neighbours = neighbours / torch.sum(neighbours)
            return_list.append(neighbours)
            data_idx += 1

        if self.use_pose:
            return_list.append(data_batch[data_idx])
            data_idx += 1

        if self.use_image_ids:
            return_list.append(data_batch[data_idx])
            data_idx += 1

        return tuple(return_list)

    def get_output_format(self):
        """
        Gets the output format
        Returns:
            list: The output format of strings

        """
        output_list = []

        if self.use_images:
            output_list.append('images')

        for r in self.rasters:
            output_list.append("feature_%s" %r)

        if self.use_categories:
            output_list.append('label')

        if self.use_neighbours:
            output_list.append('neighbours')

        if self.use_pose:
            output_list.append('pose')

        if self.use_image_ids:
            output_list.append('image_id')

        return output_list

    def cache(self):
        """
        Caches the dataset output to a directory

        Args:
            output_dir:

        Returns:

        """
        os.makedirs(self.cache_dir, exist_ok=True)
        num_iters = len(self)

        output_list = self.get_output_format()
        cache_dict = {}
        for op in output_list:
            cache_dict[op] = []

        for n, data_batch in enumerate(self):
            for op_idx, op in enumerate(cache_dict.keys()):
                cache_dict[op].append(data_batch[op_idx].detach().cpu().numpy())

            print("Caching %d/%d" %(n+1, num_iters))

        for op, data in cache_dict.items():
            fpath = os.path.join(self.cache_dir, op + ".npy")
            data_arr = np.asarray(data)
            np.save(fpath, data_arr)
        print("Finished caching")

    def print_output_format(self):
        """
        Prints the output format
        Returns:

        """
        print(','.join(self.get_output_format()))

    def load_cached(self):
        """
        Loads the cached arrays from the cached directory. If it fails to load them, will return False.
        Returns:
            bool: True if loading cached was successful, False if not.
        """
        self.cache_dict = {}
        output_list = self.get_output_format()
        for op in output_list:
            if os.path.isfile(os.path.join(self.cache_dir, op + ".npy")):
                self.cache_dict[op] = np.load(os.path.join(self.cache_dir, op + ".npy"))
            else:
                # Failed to load the cached directory
                return False
        return True

    def get_cached(self, idx):
        """
        Gets cached data
        Args:
            idx:

        Returns:
            tuple: A tuple of data corresponding to the output data.

        """
        output_list = self.get_output_format()
        return_list = []
        for op in output_list:
            return_list.append(self.cache_dict[op][idx])
        return tuple(return_list)


class LatentGenerator(data.Dataset):
    """
    Latent Generator

    General purpose latent generator, that should be compatible with any dataset configuration.
    """
    def __init__(self, dataset, encoders, cache_dir=None, cuda=True):
        self.dataset = dataset
        self.encoders = encoders
        self.rasters = self.dataset.rasters
        self.use_categories = self.dataset.use_categories
        self.use_neighbours = self.dataset.use_neighbours
        self.use_pose = self.dataset.use_pose
        self.use_image_ids = self.dataset.use_image_ids
        self.cuda = cuda
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if isinstance(self.dataset, CocoImgRaster):
            self.use_images = True
        elif isinstance(self.dataset, CocoRaster):
            self.use_images = False
        else:
            raise ValueError("BathyLatentGenerator only supports CocoImgRaster and CocoRaster data generators")

        if "bathymetry+backscatter" in self.encoders.keys() and not len(self.rasters) == 2:
            raise NotImplementedError("When used together, bathymetry and backscatter are only allowed. TODO - make other compatible")

        self.cache_dict = {}
        self.cache_dir = cache_dir
        self.cache_ok = False

        if self.cache_dir and self.load_cached():
            self.cache_ok = True
            # self.clear_encoders()
        elif self.cache_dir and not self.load_cached():
            self.cache()
            if self.load_cached():
                self.cache_ok = True
                # self.clear_encoders()
            else:
                warnings.warn("Cache not working, using live instead")

    def get_category_counts(self):
        return self.dataset.get_category_counts()

    def __len__(self):
        if self.cache_ok:  # TODO hack remove
            output_list = self.get_output_format()
            return self.cache_dict[output_list[0]].shape[0]
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets data

        Args:
            idx:

        Returns:

        """
        if self.cache_ok:
            return self.get_cached(idx)
        else:
            return self.get_live(idx)

    def clear_encoders(self):
        """
        Clears the encoders to save on cuda memory
        Returns:

        """
        for key in self.encoders.keys():
            self.encoders[key] = None

    def cache(self):
        """
        Caches the dataset output to a directory

        Args:
            output_dir:

        Returns:

        """
        os.makedirs(self.cache_dir, exist_ok=True)
        num_iters = len(self)

        output_list = self.get_output_format()
        cache_dict = {}
        for op in output_list:
            cache_dict[op] = []

        for n, data_batch in enumerate(self):
            for op_idx, op in enumerate(cache_dict.keys()):
                cache_dict[op].append(data_batch[op_idx].detach().cpu().numpy())

            print("Caching %d/%d" %(n+1, num_iters))

        for op, data in cache_dict.items():
            fpath = os.path.join(self.cache_dir, op + ".npy")
            data_arr = np.asarray(data)
            np.save(fpath, data_arr)
        print("Finished caching")

    def load_cached(self):
        """
        Loads the cached arrays from the cached directory. If it fails to load them, will return False.
        Returns:
            bool: True if loading cached was successful, False if not.
        """
        self.cache_dict = {}
        output_list = self.get_output_format()
        for op in output_list:
            if os.path.isfile(os.path.join(self.cache_dir, op + ".npy")):
                self.cache_dict[op] = np.load(os.path.join(self.cache_dir, op + ".npy"))
            else:
                # Failed to load the cached directory
                return False
        return True

    def get_cached(self, idx):
        """
        Gets cached data
        Args:
            idx:

        Returns:
            tuple: A tuple of data corresponding to the output data.

        """
        output_list = self.get_output_format()
        return_list = []
        for op in output_list:
            return_list.append(self.cache_dict[op][idx])
        return tuple(return_list)

    def get_output_format(self):
        """
        Gets the output format
        Returns:
            list: The output format of strings

        """
        output_list = []

        if self.use_images:
            output_list.append('images')

        for lt in self.encoders.keys():
            output_list.append("latent_" + lt)

        if self.use_categories:
            output_list.append('label')

        if self.use_neighbours:
            output_list.append('neighbours')

        if self.use_pose:
            output_list.append('pose')

        if self.use_image_ids:
            output_list.append('image_id')

        return output_list


    def print_output_format(self):
        """
        Prints the output format
        Returns:

        """
        print(','.join(self.get_output_format()))

class CachedLatentGeneratorNeighbours(data.Dataset):
    def __init__(self, latent_array, cat_array, neighbour_array):
        self.latent_array = latent_array
        self.cat_array = cat_array
        self.neighbour_array = neighbour_array

        # Calculate latent array
        self.dataset_length = self.latent_array.shape[0]
        if cat_array.shape[0] != self.dataset_length:
            raise ValueError("Length of Category Array is not the same as latent array")
        if self.neighbour_array.shape[0] != self.dataset_length:
            raise ValueError("Length of Neighbour Array is not the same as latent array")


    def __getitem__(self, idx):
        return self.latent_array[idx], self.cat_array[idx], self.neighbour_array[idx]

    def __len__(self):
        return self.dataset_length


def load_extractor(extractor_config, CUDA):
    if extractor_config['type'] == "bathymetry_encoder" or extractor_config['type'] == "bathy_encoder":
        if isinstance(extractor_config['params'], str):
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))
        else:
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))

        extractor = BathymetryEncoder(extractor_dictionary, extractor_config['weights'], CUDA)
    elif extractor_config['type'] == "bathymetry_vae" or extractor_config['type'] == "bathymetry_variational_encoder" or extractor_config['type'] == "bathymetry_vencoder":
        if isinstance(extractor_config['params'], str):
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))
        else:
            extractor_dictionary = json.load(open(extractor_config['params'], 'r'))

        extractor = BathymetryVariationalEncoder(extractor_dictionary, extractor_config['weights'], CUDA)
    else:
        raise ValueError("Extractor type")
    return extractor
