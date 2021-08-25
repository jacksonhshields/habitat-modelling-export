import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import json
import plyfile
import numpy as np
import utm
import math
import warnings
import random
from pycocotools.coco import COCO
import pymap3d
from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros, process_depth_map_to_array
try:
    from habitat_modelling.utils.raster_utils import extract_raster_patch, retrieve_pixel_coords, extract_geotransform
except ImportError:
    warnings.warn("Cannot import habitat_modelling.utils.raster_utils (gdal?). Live raster loading not supported")

from habitat_modelling.ml.torch.transforms.bathy_transforms import to_float_tensor, to_tensor
from habitat_modelling.ml.torch.transforms.image_transforms import image_transforms_from_cfg

class CocoImgRaster(data.Dataset):
    def __init__(self, coco_path, rasters=None, raster_paths=None, live_raster=False, raster_sizes=None,  img_transform=None, raster_transforms=None, img_dir=None, categories=False, neighbours=False, num_classes=None, pose_cfg=None, image_ids=False):
        """
        Initialises the generator

        Args:
            coco_path: (str) Path to the coco dataset
            rasters: (list) List of strings for the rasters to load. Must match elements of coco dictionary. E.g. ['bathy','backscatter']
            live_raster: (bool) Whether to extract the raster patches online. If not, the patches need to be extracted before (using dive-to-coco)
            raster_sizes: (list) [Only if live_raster] The size of the raster patches to extract
            img_transform: (torchvision.transforms) List of transform classes to apply to the images. Must have the 'transform' function.
            raster_transforms: (dict) Dictionary mapping each raster to the transform to be applied.
            img_dir: (str) Path to the image folder. Not reccomended, use 'path' in the coco datasets instead.
            categories: (bool) Whether to extract classes.
            neighbours: (bool) Whether to output the neighbour distributions
            num_classes: (int) The number of classes (if left blank, the number of classes is deduced from the category list)
            pose_cfg: (dict) The configuration that spits out the pose. If left blank, pose will not be output.
            image_ids: (bool) Whether to return the image ids corresponding that were used to extract the data.
        """
        self.coco = COCO(coco_path)
        self.rasters = rasters
        self.live_raster = live_raster
        self.raster_collection = {}
        self.raster_sizes = raster_sizes

        # Create the raster dictionary
        if self.live_raster:
            if len(raster_sizes) != len(rasters):
                raise ValueError("Length of raster list (%d) must equal the length of the raster sizes list (%d)" %(len(self.rasters), len(raster_sizes)))

            for n, rs in enumerate(self.rasters):
                # Check the raster is in the dataset
                if rs not in self.coco.dataset:
                    raise ValueError("Raster %s not in the dataset" %rs)

                if raster_paths is None:
                    rpath = self.coco.dataset[rs]['path']
                else:
                    rpath = raster_paths[n]

                raster_ds, geotransform, raster_info = extract_geotransform(rpath)
                nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()

                self.raster_collection[rs] = {
                    "dataset": raster_ds,
                    "info": raster_info,
                    "geotransform": geotransform,
                    "size": raster_sizes[n],
                    "no_data_val": nodataval
                }

        # Load all the image ids
        self.img_ids = self.coco.getImgIds()

        # Optionally use categories
        self.use_categories = categories
        if self.use_categories:
            # Load the categories
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            if num_classes:
                self.num_classes = num_classes  # In case number of classes doesn't match the number of classes in this dataset
            else:
                self.num_classes = len(self.coco.getCatIds())

        # Image Transforms
        self.img_transform = img_transform  # A torchvision transform

        # Bathy Transforms
        self.raster_transforms = raster_transforms  # A list of classes with a 'transform' function

        # Set the optional image directory - images should have 'path' in them
        self.img_dir = img_dir

        # Optionally use neighbours
        self.use_neighbours = neighbours

        # Optionally output the auv position
        if pose_cfg:
            self.use_pose = True
            if 'datum' in pose_cfg:
                self.datum = pose_cfg['datum']
            else:
                raise ValueError("Datum is needed for using pose")
            if 'use_2d' in pose_cfg and pose_cfg['use_2d']:
                self.use_2d_pose = True
            else:
                self.use_2d_pose = False
        else:
            self.use_pose = False



    def __len__(self):
        """
        Denotes the number of images in each epoch

        Returns:
            (int): number of batches

        """
        return len(self.img_ids)

    def load_raster(self, raster, image):
        """
        Loads the raster corresponding to the image
        Args:
            raster: (str) The raster format to use
            image: (dict) The coco format image entry. Needs to have geolocation or 'pose' in it for live loading. For offline loading, the path to the raster file
        Returns:
            np.ndarray: The raster at this point.
        """
        if self.live_raster:
            lla = image['geo_location']

            if 'UTM' in self.raster_collection[raster]['info']['projection']:  # TODO find less hacky way to determine
                use_utm = True
            else:
                use_utm = False

            if use_utm:
                ux, uy, zone_num, zone_letter = utm.from_latlon(lla[0], lla[1])
                px, py = retrieve_pixel_coords([ux, uy], list(self.raster_collection[raster]['geotransform']))
            else:
                px, py = retrieve_pixel_coords([lla[1], lla[0]], list(self.raster_collection[raster]['geotransform']))

            patch_size = self.raster_collection[raster]['size']

            half_patch = np.floor(patch_size[0]/2)

            off_x = int(np.round(px - half_patch))
            off_y = int(np.round(py - half_patch))

            patch = extract_raster_patch(self.raster_collection[raster]['dataset'], off_x, off_y, patch_size[0])
        else:
            patch = np.load(image[raster]['path'])

        if np.any(patch == self.raster_collection[raster]["no_data_val"]):
            patch = None

        if 'bathy' in raster: #raster == "bathy" or raster == "bathymetry":  # Separate this, as the depth needs to be separated
            if patch is None:
                return None, None
            centre_depth = patch[round(patch.shape[0]/2), round(patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            return dmap, mean_depth
        else:
            return patch

    def get_pose(self, image):
        lla = image['geo_location']
        nx, ey, dz = pymap3d.geodetic2ned(lla[0], lla[1], lla[2], self.datum[0], self.datum[1], self.datum[2])
        if self.use_2d_pose:
            return np.array([nx, ey])
        else:
            return np.array([nx,ey,dz])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_idx = index  # TODO store this?
        img_id = self.img_ids[img_idx]

        not_ok_count = 0

        data_collected = False

        data_dict = {}
        data_dict['image'] = None
        for r in self.rasters:
            if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                data_dict['dmap'] = None
                data_dict['depth'] = None
            else:
                data_dict[r] = None
        if self.use_categories:
            data_dict['categories'] = None

        # Spin until data from each source is detected
        while not data_collected:
            img_id = self.img_ids[img_idx]

            data_dict = {}
            data_dict['image'] = None
            for r in self.rasters:
                if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                    data_dict['dmap'] = None
                    data_dict['depth'] = None
                else:
                    data_dict[r] = None
            if self.use_categories:
                data_dict['categories'] = None

            image = self.coco.loadImgs([img_id])[0]

            # Select the image path - should usually use the path key
            if self.img_dir:  # Not reccomended
                img_path = os.path.join(self.img_dir, image['file_name'])
            elif 'path' in image:
                img_path = image['path']
            else:
                # This will probably fail
                img_path = image['file_name']


            # Load the image
            pimg = Image.open(img_path).convert("RGB")
            img = pimg

            # Load each raster and allocate them to the dictionary
            raster_patches = {}
            for r in self.rasters:
                raster_patches[r] = self.load_raster(raster=r, image=image)

            # Transforms the image
            if self.img_transform:
                img = self.img_transform(img)
                # Should be a Tensor after this
            timg = img


            # Extract the categories. For these datasets, the annotation is a single point label
            if self.use_categories:
                # Get annotations - should be just one per image
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
                if len(annotations) > 0:
                    cat = annotations[0]['category_id']  # Just get first annotation. Categories are indexed.
                else:
                    cat = None
            else:
                cat = None

            if self.use_neighbours:
                # Get annotations - should be just one per image
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
                if len(annotations) > 0:
                    if 'neighbours' in annotations[0]:
                        neighbours = annotations[0]['neighbours']
                    else:
                        neighbours = None
                else:
                    cat = None

            # Check all data is collected properly

            # Check the image
            img_ok = True
            if img is None:
                img_ok = False
            # Check rasters
            bad_raster = False # TODO remove this, temporary
            rasters_ok = True  # This will be set False if any raster is not OK.
            for r,patch in raster_patches.items():
                if 'bathy' in r:# r == 'bathy' or r == 'bathymetry':
                    if patch[0] is None or patch[1] is None:
                        rasters_ok = True # TODO make false
                        dm = np.zeros(self.raster_sizes[0])  # TODO remove this
                        depth = np.array([0.0])  # TODO remove this
                        bad_raster=True # TODO remove this
                        raster_patches[r] = (dm, depth)
                    if np.isnan(patch[1]) or np.isnan(patch[0]).any():
                        dm = np.nan_to_num(patch[0], copy=True)
                        dep = np.nan_to_num(patch[1], copy=True)
                        raster_patches[r] = (dm, dep)
                        rasters_ok = True
                else:
                    if patch is None:
                        rasters_ok = False
                        continue
                    if np.isnan(patch).any():
                        rasters_ok = False

            # Transform the raster patches
            if self.raster_transforms and rasters_ok:
                for r, patch in raster_patches.items():
                    if patch is None:
                        continue
                    if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                        if patch[0] is None or patch[1] is None:
                            continue
                        dmap = patch[0] # In this case patch is a tuple (dmap, depth)
                        # Only operate on the depth map
                        dmap = self.raster_transforms[r](dmap)
                        # Add the patch tuple back in
                        raster_patches[r] = (dmap, patch[1])
                    else:
                        # Transform the patch
                        patch = self.raster_transforms[r](patch)
                        # Transform the patch
                        raster_patches[r] = patch

            cat_ok = True
            # Check the categories
            if self.use_categories and cat is None:
                cat_ok = False

            neighbours_ok = True
            # Check the neighbours
            if self.use_neighbours and neighbours is None:
                neighbours_ok = False

            img_idx += 1
            if img_idx >= len(self.img_ids):
                img_idx = 0

            pose_ok = True
            if self.use_pose:
                pos = self.get_pose(image)


            if img_ok and rasters_ok and cat_ok and neighbours_ok and pose_ok:  # Check that all patches have been extracted
                data_collected = True

        return_list = []
        return_list.append(img)

        for r, patch in raster_patches.items():
            if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                dmap = patch[0]
                depth = patch[1]
                if len(dmap.shape) == 2:
                    dmap = dmap.unsqueeze(0)
                return_list.append(dmap)
                return_list.append(depth)
            else:
                if len(patch.shape) == 2:
                    patch = patch.unsqueeze(0)
                return_list.append(patch)

        if self.use_categories:
            onehot = np.zeros([self.num_classes])
            onehot[cat] = 1.0
            if bad_raster:
                onehot = np.zeros([self.num_classes])
                bad_raster = False
            return_list.append(torch.tensor(onehot))

        if self.use_neighbours:
            return_list.append(torch.tensor(neighbours))

        if self.use_pose:
            return_list.append(torch.tensor(pos))

        if self.use_image_ids:
            return_list.append(torch.tensor(img_id))


        return return_list

    def print_output_format(self):
        print_list = []
        print_list.append('images')
        for r in self.rasters:
            if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                print_list.append('dmap')
                print_list.append('depth')
            else:
                print_list.append(r)
        if self.use_categories:
            print_list.append('label')

        if self.use_neighbours:
            print_list.append('neighbours')

        if self.use_pose:
            print_list.append('pose')

        if self.use_image_ids:
            print_list.append('image_id')

        print(','.join(print_list))

class NotRelevant(data.Dataset):
    def __init__(self, coco_path, raster_config, extractor_config, image_config=None, categories=False, neighbours=False, num_classes=None, pose_cfg=None, image_ids=False, position_variance=None, CUDA=True):
        self.coco = COCO(coco_path)
        self.raster_registry = self._init_rasters(raster_config)
        self.extractors = self._init_extractors(extractor_config, CUDA)
        self.CUDA = CUDA
        self.use_images, self.img_transforms = self._init_images(image_config)
        # Load all the image ids
        self.img_ids = self.coco.getImgIds()
        # Optionally use categories
        self.use_categories = categories
        if self.use_categories:
            # Load the categories
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            if num_classes:
                self.num_classes = num_classes  # In case number of classes doesn't match the number of classes in this dataset
            else:
                self.num_classes = len(self.coco.getCatIds())
        # Optionally use neighbours
        self.use_neighbours = neighbours
        # Optionally, use the image ids.
        self.use_image_ids = image_ids

        # Optionally output the auv position
        if pose_cfg:
            self.use_pose = True
            if 'datum' in pose_cfg:
                self.datum = pose_cfg['datum']
            else:
                raise ValueError("Datum is needed for using pose")
            if 'use_2d' in pose_cfg and pose_cfg['use_2d']:
                self.use_2d_pose = True
            else:
                self.use_2d_pose = False
        else:
            self.use_pose = False

        if position_variance:
            self.use_position_variance = True
            self.position_variance = position_variance
            # Create datum
            self.datum = self.raster_registry[self.raster_registry.keys()[0]]['origin'] + [0.]
            # TODO make centre pixel?
        else:
            self.use_position_variance= False


    def _init_rasters(self, raster_config):
        raster_registry = {}
        for key, entry in raster_config:
            raster_ds, geotransform, raster_info = extract_geotransform(entry['path'])
            nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()

            if 'transforms' in entry:
                transforms = []
                for t in entry['transforms']:
                    if t == "tensor":
                        transforms.append(to_tensor)
                    elif t == 'float_tensor':
                        transforms.append(to_float_tensor)
            else:
                transforms = []

            raster_registry[key] = {
                "dataset": raster_ds,
                "info": raster_info,
                "geotransform": geotransform,
                "size": entry['size'],
                "no_data_val": nodataval,
                "transforms": transforms
            }

        return raster_registry

    def _init_extractors(self, extractor_config, CUDA):
        from habitat_modelling.methods.latent_mapping.feature_extractors import load_extractor
        extractors = []
        for ext_cfg in extractor_config:
            ext = {
                "inputs": ext_cfg["inputs"],
                "model": load_extractor(ext_cfg)
            }
            extractors.append(ext)
        return extractors

    def _init_images(self, image_config):
        if image_config is None:
            return (False, None)
        if 'transforms' in image_config:
            img_transforms = image_transforms_from_cfg(image_config['transforms'])
        else:
            img_transforms = None
        return True, img_transforms

    def get_output_format(self):
        print_list = []
        for ex in self.extractors:
            print_list.append('+'.join(ex['inputs']))
            if any('bathy' in r for r in ex['inputs']):
                print_list.append('depth')
        if self.use_categories:
            print_list.append('label')
        if self.use_neighbours:
            print_list.append('neighbours')
        if self.use_pose:
            print_list.append('pose')
        if self.use_image_ids:
            print_list.append('image_id')
        return print_list

    def print_output_format(self):
        print_list = self.get_output_format()
        print(','.join(print_list))

    def get_pose(self, image):
        lla = image['geo_location']
        nx, ey, dz = pymap3d.geodetic2ned(lla[0], lla[1], lla[2], self.datum[0], self.datum[1], self.datum[2])
        if self.use_2d_pose:
            return np.array([nx, ey])
        else:
            return np.array([nx,ey,dz])

    def lla_from_pose(self, pose):
        """
        Gets LLA from Pose using the datum
        Args:
            pose: (np.ndarray) pose in format (nx,ey,dz)
        Returns:
            lla: lat, lon, alt
        """
        lla = pymap3d.ned2geodetic(pose[0], pose[1], pose[2], self.datum[0], self.datum[1], self.datum[2])
        return lla

    def add_variance(self, image):
        pose = self.get_pose(image)
        pose[0] = pose[0] + np.random.normal(0, np.sqrt(self.position_variance[0]))
        pose[1] = pose[1] + np.random.normal(0, np.sqrt(self.position_variance[1]))
        lla = self.lla_from_pose(pose)
        return lla

    def load_patch(self, raster, image):
        """
        Loads the raster corresponding to the image
        Args:
            raster: (str) The raster format to use
            image: (dict) The coco format image entry. Needs to have geolocation or 'pose' in it for live loading. For offline loading, the path to the raster file
        Returns:
            np.ndarray: The raster at this point.
        """
        if self.live_raster:
            lla = image['geo_location']

            if self.use_position_variance:
                lla = self.add_variance(image)

            if 'UTM' in self.raster_collection[raster]['info']['projection']:  # TODO find less hacky way to determine
                use_utm = True
            else:
                use_utm = False

            if use_utm:
                ux, uy, zone_num, zone_letter = utm.from_latlon(lla[0], lla[1])
                px, py = retrieve_pixel_coords([ux, uy], list(self.raster_collection[raster]['geotransform']))
            else:
                px, py = retrieve_pixel_coords([lla[1], lla[0]], list(self.raster_collection[raster]['geotransform']))

            patch_size = self.raster_collection[raster]['size']

            half_patch = np.floor(patch_size[0]/2)

            off_x = int(np.round(px - half_patch))
            off_y = int(np.round(py - half_patch))

            patch = extract_raster_patch(self.raster_collection[raster]['dataset'], off_x, off_y, patch_size[0])
        else:
            patch = np.load(image[raster]['path'])

        # Check for no data values
        # if np.any(patch == self.no_data_values[raster]):
        #     patch = None  # TODO make it work
        if np.any(patch == self.raster_collection[raster]["no_data_val"]):
            patch = None

        if "bathy" in raster: #raster == "bathy" or raster == "bathymetry":  # Separate this, as the depth needs to be separated
            if patch is None:
                return None, None
            centre_depth = patch[round(patch.shape[0]/2), round(patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            return dmap, mean_depth
        else:
            return patch, None

    def load_image(self, image):
        """
        Loads and transforms and image
        Args:
            image: (dict) The coco image entry. Needs to have 'path'
        Returns:
            (torch.tensor) The transformed image as a tensor
        """
        img_path = image['path']
        # Load the image
        pimg = Image.open(img_path).convert("RGB")
        img = pimg
        # Transforms the image
        if self.img_transforms:
            img = self.img_transforms(img)
            # Should be a Tensor after this
        timg = img
        return timg

    def load_features(self, image, extractor):
        for raster_name in extractor['inputs']:
            patch, depth = self.load_patch(image)
            if 'transform' in self.raster_registry[raster_name] and self.raster_registry[raster_name]['transform'] is not None:
                patch_t = self.raster_registry[raster_name]['transform'].transform(patch)
            else:
                patch_t = patch
            features = self.extractors(patch_t)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_idx = index  # TODO store this?
        img_id = self.img_ids[img_idx]
        data_collected = False

        while not data_collected:
            img_id = self.img_ids[img_idx]
            image = self.coco.loadImgs([img_id])[0]
            if self.use_images:
                timg = self.load_image(image)
            else:
                timg = None

            for ex in self.extractors:
                pass
















class CocoRaster(data.Dataset):
    def __init__(self, coco_path, rasters=None, raster_paths=None, live_raster=False, raster_sizes=None,  img_transform=None, raster_transforms=None, img_dir=None, categories=False, neighbours=False, num_classes=None, pose_cfg=None, image_ids=False, position_variance=None):
        """
        Initialises the generator

        Args:
            coco_path: (str) Path to the coco dataset
            rasters: (list) List of strings for the rasters to load. Must match elements of coco dictionary. E.g. ['bathy','backscatter']
            raster_paths: (list) An optional dictionary of raster paths. If not given, the coco will determine the rasters
            live_raster: (bool) Whether to extract the raster patches online. If not, the patches need to be extracted before (using dive-to-coco)
            raster_sizes: (list) [Only if live_raster] The size of the raster patches to extract
            img_transform: (torchvision.transforms) List of transform classes to apply to the images. Must have the 'transform' function.
            raster_transforms: (dict) Dictionary mapping each raster to the transform to be applied.
            img_dir: (str) Path to the image folder. Not reccomended, use 'path' in the coco datasets instead.
            categories: (bool) Whether to extract classes.
            neighbours: (bool) Whether to output the neighbour distributions
            num_classes: (int) The number of classes (if left blank, the number of classes is deduced from the category list)
            pose_cfg: (dict) The configuration that spits out the pose. If left blank, pose will not be output.
            image_ids: (bool) Whether to return the image ids corresponding that were used to extract the data.
        """
        self.coco = COCO(coco_path)
        self.rasters = rasters
        self.live_raster = live_raster
        self.raster_collection = {}
        self.raster_sizes = raster_sizes

        # Create the raster dictionary
        if self.live_raster:
            if len(raster_sizes) != len(rasters):
                raise ValueError("Length of raster list (%d) must equal the length of the raster sizes list (%d)" %(len(self.rasters), len(raster_sizes)))

            for n, rs in enumerate(self.rasters):
                # Check the raster is in the dataset
                if raster_paths is None:
                    if rs not in self.coco.dataset:
                        raise ValueError("Raster %s not in the dataset" % rs)
                    rpath = self.coco.dataset[rs]['path']
                else:
                    rpath = raster_paths[n]
                raster_ds, geotransform, raster_info = extract_geotransform(rpath)
                nodataval = raster_ds.GetRasterBand(1).GetNoDataValue()

                self.raster_collection[rs] = {
                    "dataset": raster_ds,
                    "info": raster_info,
                    "geotransform": geotransform,
                    "size": raster_sizes[n],
                    "no_data_val": nodataval
                }

        # Load all the image ids
        self.img_ids = self.coco.getImgIds()

        # self.no_data_values=no_data_values
        # self.no_data_values = {
        #     'bathymetry': -9999,
        #     'backscatter': 0
        # }

        # Optionally use categories
        self.use_categories = categories
        if self.use_categories:
            # Load the categories
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            if num_classes:
                self.num_classes = num_classes  # In case number of classes doesn't match the number of classes in this dataset
            else:
                self.num_classes = len(self.coco.getCatIds())

        # Image Transforms
        self.img_transform = img_transform  # A torchvision transform

        # Bathy Transforms
        self.raster_transforms = raster_transforms  # A list of classes with a 'transform' function

        # Set the optional image directory - images should have 'path' in them
        self.img_dir = img_dir

        # Optionally use neighbours
        self.use_neighbours = neighbours

        # Optionally output the auv position
        if pose_cfg:
            self.use_pose = True
            if 'datum' in pose_cfg:
                self.datum = pose_cfg['datum']
            else:
                raise ValueError("Datum is needed for using pose")
            if 'use_2d' in pose_cfg and pose_cfg['use_2d']:
                self.use_2d_pose = True
            else:
                self.use_2d_pose = False
        else:
            self.use_pose = False
            self.use_2d_pose = False

        # Optionally, use the image ids.
        self.use_image_ids = image_ids

        if position_variance:
            self.use_position_variance = True
            self.position_variance = position_variance
            # Create datum
            self.datum = raster_info['origin'] + [0.]
            # TODO make centre pixel?
        else:
            self.use_position_variance= False

    def __len__(self):
        """
        Denotes the number of images in each epoch

        Returns:
            (int): number of batches

        """
        return len(self.img_ids)

    def get_pose(self, image):
        lla = image['geo_location']
        nx, ey, dz = pymap3d.geodetic2ned(lla[0], lla[1], lla[2], self.datum[0], self.datum[1], self.datum[2])
        if self.use_2d_pose:
            return np.array([nx, ey])
        else:
            return np.array([nx,ey,dz])

    def lla_from_pose(self, pose):
        """
        Gets LLA from Pose using the datum
        Args:
            pose: (np.ndarray) pose in format (nx,ey,dz)
        Returns:
            lla: lat, lon, alt
        """
        lla = pymap3d.ned2geodetic(pose[0], pose[1], pose[2], self.datum[0], self.datum[1], self.datum[2])
        return lla

    def add_variance(self, image):
        pose = self.get_pose(image)
        pose[0] = pose[0] + np.random.normal(0, np.sqrt(self.position_variance[0]))
        pose[1] = pose[1] + np.random.normal(0, np.sqrt(self.position_variance[1]))
        lla = self.lla_from_pose(pose)
        return lla

    def load_raster(self, raster, image):
        """
        Loads the raster corresponding to the image
        Args:
            raster: (str) The raster format to use
            image: (dict) The coco format image entry. Needs to have geolocation or 'pose' in it for live loading. For offline loading, the path to the raster file
        Returns:
            np.ndarray: The raster at this point.
        """
        if self.live_raster:
            lla = image['geo_location']

            if self.use_position_variance:
                lla = self.add_variance(image)

            if 'UTM' in self.raster_collection[raster]['info']['projection']:  # TODO find less hacky way to determine
                use_utm = True
            else:
                use_utm = False

            if use_utm:
                ux, uy, zone_num, zone_letter = utm.from_latlon(lla[0], lla[1])
                px, py = retrieve_pixel_coords([ux, uy], list(self.raster_collection[raster]['geotransform']))
            else:
                px, py = retrieve_pixel_coords([lla[1], lla[0]], list(self.raster_collection[raster]['geotransform']))

            patch_size = self.raster_collection[raster]['size']

            half_patch = np.floor(patch_size[0]/2)

            off_x = int(np.round(px - half_patch))
            off_y = int(np.round(py - half_patch))

            patch = extract_raster_patch(self.raster_collection[raster]['dataset'], off_x, off_y, patch_size[0])
        else:
            patch = np.load(image[raster]['path'])

        # Check for no data values
        # if np.any(patch == self.no_data_values[raster]):
        #     patch = None  # TODO make it work
        if np.any(patch == self.raster_collection[raster]["no_data_val"]):
            patch = None

        if "bathy" in raster: #raster == "bathy" or raster == "bathymetry":  # Separate this, as the depth needs to be separated
            if patch is None:
                return None, None
            centre_depth = patch[round(patch.shape[0]/2), round(patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            return dmap, mean_depth
        else:
            return patch

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_idx = index  # TODO store this?
        img_id = self.img_ids[img_idx]

        not_ok_count = 0

        data_collected = False

        data_dict = {}
        data_dict['image'] = None
        for r in self.rasters:
            if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                data_dict['dmap'] = None
                data_dict['depth'] = None
            else:
                data_dict[r] = None
        if self.use_categories:
            data_dict['categories'] = None

        # Spin until data from each source is detected
        while not data_collected:
            img_id = self.img_ids[img_idx]

            image = self.coco.loadImgs([img_id])[0]

            data_dict = {}
            for r in self.rasters:
                if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                    data_dict['dmap'] = None
                    data_dict['depth'] = None
                else:
                    data_dict[r] = None
            if self.use_categories:
                data_dict['categories'] = None

            # Load each raster and allocate them to the dictionary
            raster_patches = {}
            for r in self.rasters:
                raster_patches[r] = self.load_raster(raster=r, image=image)




            # Extract the categories. For these datasets, the annotation is a single point label
            if self.use_categories:
                # Get annotations - should be just one per image
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
                if len(annotations) > 0:
                    cat = annotations[0]['category_id']  # Just get first annotation. Categories are indexed.
                else:
                    cat = None
            else:
                cat = None

            if self.use_neighbours:
                # Get annotations - should be just one per image
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
                if len(annotations) > 0:
                    if 'neighbours' in annotations[0]:
                        neighbours = annotations[0]['neighbours']
                    else:
                        neighbours = None
                else:
                    cat = None

            # Check all data is collected properly


            # Check rasters
            bad_raster = False # TODO remove this, temporary
            rasters_ok = True  # This will be set False if any raster is not OK.

            for r,patch in raster_patches.items():
                if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                    if patch[0] is None or patch[1] is None:
                        dm = np.zeros(self.raster_sizes[0])  # TODO remove this
                        depth = np.array([0.0])  # TODO remove this
                        bad_raster=True # TODO remove this
                        raster_patches[r] = (dm, depth)
                        rasters_ok = False
                    elif np.isnan(patch[1]) or np.isnan(patch[0]).any():
                        dm = np.nan_to_num(patch[0], copy=True)
                        dep = np.nan_to_num(patch[1], copy=True)
                        raster_patches[r] = (dm, dep)
                        rasters_ok = True
                else:
                    if patch is None:
                        rasters_ok = False
                        continue
                    if np.isnan(patch).any():
                        rasters_ok = False

            # Transform the raster patches
            if self.raster_transforms and rasters_ok:
                for r, patch in raster_patches.items():
                    if patch is None:
                        continue
                    if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                        if patch[0] is None or patch[1] is None:
                            continue
                        dmap = patch[0] # In this case patch is a tuple (dmap, depth)
                        # Only operate on the depth map
                        dmap = self.raster_transforms[r](dmap)
                        # Add the patch tuple back in
                        raster_patches[r] = (dmap, patch[1])
                    else:
                        # Transform the patch
                        patch = self.raster_transforms[r](patch)
                        # Transform the patch
                        raster_patches[r] = patch

            cat_ok = True
            # Check the categories
            if self.use_categories and cat is None:
                cat_ok = False

            neighbours_ok = True
            # Check the neighbours
            if self.use_neighbours and neighbours is None:
                neighbours_ok = False


            pose_ok = True
            if self.use_pose:
                pos = self.get_pose(image)

            img_idx += 1
            if img_idx >= len(self.img_ids):
                img_idx = 0

            if rasters_ok and cat_ok and neighbours_ok and pose_ok:  # Check that all patches have been extracted
                data_collected = True

        return_list = []

        for r, patch in raster_patches.items():
            if 'bathy' in r: # r == 'bathy' or r == 'bathymetry':
                dmap = patch[0]
                depth = patch[1]
                if len(dmap.shape) == 2:
                    dmap = dmap.unsqueeze(0)
                return_list.append(dmap)
                return_list.append(depth)
            else:
                if len(patch.shape) == 2:
                    patch = patch.unsqueeze(0)
                return_list.append(patch)

        if self.use_categories:
            onehot = np.zeros([self.num_classes])
            onehot[cat] = 1.0
            if bad_raster:
                onehot = np.zeros([self.num_classes])
                bad_raster = False
            return_list.append(torch.tensor(onehot))

        if self.use_neighbours:
            return_list.append(torch.tensor(neighbours))

        if self.use_pose:
            return_list.append(torch.tensor(pos))

        if self.use_image_ids:
            return_list.append(torch.tensor(img_id))


        return return_list

    def get_output_format(self):
        print_list = []
        for r in self.rasters:
            if 'bathy' in r: #r == 'bathy' or r == 'bathymetry':
                print_list.append('dmap')
                print_list.append('depth')
            else:
                print_list.append(r)
        if self.use_categories:
            print_list.append('label')

        if self.use_neighbours:
            print_list.append('neighbours')

        if self.use_pose:
            print_list.append('pose')

        if self.use_image_ids:
            print_list.append('image_id')

        return print_list


    def print_output_format(self):
        print_list = self.get_output_format()

        print(','.join(print_list))

    def get_category_counts(self):
        cids = self.coco.getCatIds()
        if max(cids) >= len(cids):
            warnings.warn("Categories not in order")
            return None
        counts = np.zeros(len(cids))
        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            cid = ann['category_id']
            if cid in cids:
                counts[cid] += 1
            else:
                return None
        return counts


class CocoImgBathy(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, annFile, root=None, img_transform=None, dmap_transform=None, depth_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.img_transform = img_transform
        self.dmap_transform = dmap_transform
        self.depth_transform = depth_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]

        if self.root is not None:
            im_fname = coco.loadImgs(img_id)[0]['file_name']
            im_path = os.path.join(self.root, im_fname)
            bathy_fname = coco.loadImgs(img_id)[0]['bathy']['file_name']
            bathy_path = os.path.join(self.root, bathy_fname)
        else:
            im_path = coco.loadImgs(img_id)[0]['path']
            bathy_path = coco.loadImgs(img_id)[0]['bathy']['path']

        img = Image.open(im_path).convert('RGB')

        bathy_patch = np.load(bathy_path)

        if bathy_patch is None:
            return img, None

        # Remove the zeros, replace with mean
        dmap = remove_depth_map_zeros(bathy_patch)
        dmap, mean_depth = make_depth_map_mean_zero(dmap)
        # TODO use centre or mean depth??
        centre_depth = bathy_patch[round(bathy_patch.shape[0] / 2), round(bathy_patch.shape[1] / 2)]


        # Needs to be in PIL format to transform (e.g. into tensor, centre crop, resize)
        # dmapil = Image.fromarray(dmap.astype(np.float32), mode='F')

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.dmap_transform is not None:
            dmap = self.dmap_transform(dmap)
        if len(dmap.shape) == 2:
            dmap = np.expand_dims(dmap, axis=0)
        dmap = torch.tensor(dmap)

        mean_depth = torch.tensor(mean_depth)

        return img, dmap, mean_depth

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Img Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.img_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Dmap Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dmap_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CocoImgMesh(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, annFile, root=None, img_transform=None, dmap_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.img_transform = img_transform
        self.dmap_transform = dmap_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]

        if self.root is not None:
            im_fname = coco.loadImgs(img_id)[0]['file_name']
            im_path = os.path.join(self.root, im_fname)
            mesh_fname = coco.loadImgs(img_id)[0]['mesh']['file_name']
            mesh_path = os.path.join(self.root, mesh_fname)
        else:
            im_path = coco.loadImgs(img_id)[0]['path']
            mesh_path = coco.loadImgs(img_id)[0]['mesh']['path']

        img = Image.open(im_path).convert('RGB')
        ply_data = plyfile.PlyData.read(mesh_path)

        # Create a depth map from the ply data
        depth_map = ply_to_depth_map(ply_data, resolution=0.1)
        # Remove the zeros, replace with mean
        dmap = process_depth_map_to_array(depth_map)

        dmapil = Image.fromarray(dmap.astype(np.float32), mode='F')

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.dmap_transform is not None:
            dmapil = self.dmap_transform(dmapil)

        return img, dmapil

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Img Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.img_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Dmap Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dmap_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
