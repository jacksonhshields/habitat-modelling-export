import keras
import numpy as np
from pycocotools.coco import COCO
import random
from PIL import Image
import os
import utm
import warnings

from habitat_modelling.core.depth_map import make_depth_map_mean_zero, remove_depth_map_zeros
try:
    from habitat_modelling.utils.raster_utils import extract_raster_patch, retrieve_pixel_coords, extract_geotransform
except ImportError:
    warnings.warn("Cannot import habitat_modelling.utils.raster_utils (gdal?). Live raster loading not supported")

class CocoImgBathyGenerator(keras.utils.Sequence):
    def __init__(self, coco_path, batch_size=1, shuffle=True, img_transforms=None, bathy_transforms=None, img_dir=None, bathy_dir=None):
        self.coco = COCO(coco_path)

        self.batch_size = batch_size

        # Load all the image ids
        self.img_ids = self.coco.getImgIds()

        # Load the categories
        # self.categories = self.coco.loadCats(self.coco.getCatIds())

        # Whether to shuffle the dataset
        self.shuffle = shuffle

        # Image Transforms
        self.img_transforms = img_transforms  # A list of classes with a 'transform' function

        # Bathy Transforms
        self.bathy_transforms = bathy_transforms  # A list of classes with a 'transform' function

        # Set the optional image directory - images should have 'path' in them
        self.img_dir = img_dir
        self.bathy_dir = bathy_dir
        if self.img_dir and not self.bathy_dir:
            raise ValueError("If img_dir is given (not recommended), then bathy_dir needs to be given as well")
        if self.bathy_dir and not self.img_dir:
            raise ValueError("If bathy_dir is given (not recommended), then img_dir needs to be given as well")

        if self.shuffle:
            # This shuffles in place
            random.shuffle(self.img_ids)

    def __len__(self):
        """
        Denotes the number of batches in an epoch

        Returns:
            (int): number of batches

        """
        return int(np.floor(len(self.img_ids)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.img_ids)

    def __getitem__(self, index):
        """
        Generates a batch of data
        Args:
            index:

        Returns:

        """
        # Gets the img_ids for this batch
        image_batch_ids = self.img_ids[index*self.batch_size:(index+1)*self.batch_size]

        # Get the images for this batch
        images = self.coco.loadImgs(image_batch_ids)

        # image/images: a dict/list of dicts containing image information according to coco standard
        # img/imgs: actual img (e.g. np array)

        imgs_list = []
        bathy_list = []
        depth_list = []


        for image in images:
            if self.img_dir:  # Not reccomended
                img_path = os.path.join(self.img_dir, image['file_name'])
                bathy_path = os.path.join(self.bathy_dir, image['bathy']['file_name'])
            elif 'path' in image:
                img_path = image['path']
                bathy_path = image['bathy']['path']
            else:
                # This will probably fail
                img_path = image['file_name']
                bathy_path = image['bathy']['file_name']

            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)
            bathy_patch = np.load(bathy_path)

            centre_depth = bathy_patch[round(bathy_patch.shape[0]/2), round(bathy_patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(bathy_patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            # Transforms

            if self.img_transforms:
                for trans in self.img_transforms:
                    img = trans.transform(img)

            if self.bathy_transforms:
                for trans in self.bathy_transforms:
                    dmap = trans.transform(dmap)



            imgs_list.append(img)
            bathy_list.append(dmap)
            depth_list.append(mean_depth)


        bathy_array = np.asarray(bathy_list)
        # Try to deal with encoding of bathymetry...
        # If bathymetry is in image format - expand dimensions
        if len(bathy_array.shape) >= 3:
            bathy_array = np.expand_dims(np.asarray(bathy_list), axis=3)


        return np.asarray(imgs_list), bathy_array, np.asarray(depth_list)




class CocoImgBathyCatGenerator(keras.utils.Sequence):
    def __init__(self, coco_path, batch_size=1, shuffle=True, img_transforms=None, bathy_transforms=None, img_dir=None, bathy_dir=None):
        self.coco = COCO(coco_path)

        self.batch_size = batch_size

        # Load all the image ids
        self.img_ids = self.coco.getImgIds()

        # Load the categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())

        self.num_classes = len(self.coco.getCatIds())

        # Whether to shuffle the dataset
        self.shuffle = shuffle

        # Image Transforms
        self.img_transforms = img_transforms  # A list of classes with a 'transform' function

        # Bathy Transforms
        self.bathy_transforms = bathy_transforms  # A list of classes with a 'transform' function

        # Set the optional image directory - images should have 'path' in them
        self.img_dir = img_dir
        self.bathy_dir = bathy_dir
        if self.img_dir and not self.bathy_dir:
            raise ValueError("If img_dir is given (not recommended), then bathy_dir needs to be given as well")
        if self.bathy_dir and not self.img_dir:
            raise ValueError("If bathy_dir is given (not recommended), then img_dir needs to be given as well")

        if self.shuffle:
            # This shuffles in place
            random.shuffle(self.img_ids)

    def __len__(self):
        """
        Denotes the number of batches in an epoch

        Returns:
            (int): number of batches

        """
        return int(np.floor(len(self.img_ids)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.img_ids)

    def __getitem__(self, index):
        """
        Generates a batch of data
        Args:
            index:

        Returns:

        """
        # Gets the img_ids for this batch
        image_batch_ids = self.img_ids[index*self.batch_size:(index+1)*self.batch_size]

        # Get the images for this batch
        images = self.coco.loadImgs(image_batch_ids)

        # image/images: a dict/list of dicts containing image information according to coco standard
        # img/imgs: actual img (e.g. np array)

        imgs_list = []
        bathy_list = []
        depth_list = []
        category_list = []

        img_idx = self.img_ids[index]

        while len(imgs_list) < self.batch_size:

            image = self.coco.loadImgs([img_idx])[0]

            if self.img_dir:  # Not reccomended
                img_path = os.path.join(self.img_dir, image['file_name'])
                bathy_path = os.path.join(self.bathy_dir, image['bathy']['file_name'])
            elif 'path' in image:
                img_path = image['path']
                bathy_path = image['bathy']['path']
            else:
                # This will probably fail
                img_path = image['file_name']
                bathy_path = image['bathy']['file_name']

            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)
            bathy_patch = np.load(bathy_path)

            centre_depth = bathy_patch[round(bathy_patch.shape[0]/2), round(bathy_patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(bathy_patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            # Transforms

            if self.img_transforms:
                for trans in self.img_transforms:
                    img = trans.transform(img)

            if self.bathy_transforms:
                for trans in self.bathy_transforms:
                    dmap = trans.transform(dmap)

            # Get annotations - should be just one per image
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
            if len(annotations) > 0:
                cat = annotations[0]['category_id']  # Just get first annotation
                category_list.append(cat-1)  # Categories are indexed from 1.
                imgs_list.append(img)
                bathy_list.append(dmap)
                depth_list.append(mean_depth)

            # Increment image index
            img_idx += 1
            if img_idx >= len(self.img_ids):  # If img idx exceeds
                img_idx = 0


        bathy_array = np.asarray(bathy_list)
        # Try to deal with encoding of bathymetry...
        # If bathymetry is in image format - expand dimensions
        if len(bathy_array.shape) >= 3:
            bathy_array = np.expand_dims(np.asarray(bathy_list), axis=3)


        return np.asarray(imgs_list), bathy_array, np.asarray(depth_list), keras.utils.to_categorical(category_list, num_classes=len(self.coco.getCatIds()))



class CocoImgRasterGenerator(keras.utils.Sequence):
    def __init__(self, coco_path, rasters=None, batch_size=1, shuffle=True, live_raster=False, raster_sizes=None,  img_transforms=None, raster_transforms=None, img_dir=None, categories=False):
        """
        Initialises the generator

        Args:
            coco_path: (str) Path to the coco dataset
            rasters: (list) List of strings for the rasters to load. Must match elements of coco dictionary. E.g. ['bathy','backscatter']
            batch_size: (int) The batch size
            shuffle: (bool) Whether to shuffle the dataset
            live_raster: (bool) Whether to extract the raster patches online. If not, the patches need to be extracted before (using dive-to-coco)
            raster_sizes: (list) [Only if live_raster] The size of the raster patches to extract
            img_transforms: (list) List of transform classes to apply to the images. Must have the 'transform' function.
            raster_transforms: (dict) Dictionary mapping each raster to the transform list to be applied. Each list is a list of classes with the 'transform' attribute.
            img_dir: (str) Path to the image folder. Not reccomended, use 'path' in the coco datasets instead.
            categories: (bool) Whether to extract classes.
        """
        self.coco = COCO(coco_path)
        self.rasters = rasters
        self.live_raster = live_raster
        self.raster_collection = {}

        # Create the raster dictionary
        if self.live_raster:
            if len(raster_sizes) != len(rasters):
                raise ValueError("Length of raster list (%d) must equal the length of the raster sizes list (%d)" %(len(self.rasters), len(raster_sizes)))

            for n, rs in enumerate(self.rasters):
                # Check the raster is in the dataset
                if rs not in self.coco.dataset:
                    raise ValueError("Raster %s not in the dataset" %rs)

                raster_ds, geotransform, raster_info = extract_geotransform(self.coco.dataset[rs]['path'])
                self.raster_collection[rs] = {
                    "dataset": raster_ds,
                    "info": raster_info,
                    "geotransform": geotransform,
                    "size": raster_sizes[n]
                }

        self.batch_size = batch_size

        # Load all the image ids
        self.img_ids = self.coco.getImgIds()


        # Optionally use categories
        self.use_categories = categories
        if self.use_categories:
            # Load the categories
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            self.num_classes = len(self.coco.getCatIds())

        # Whether to shuffle the dataset
        self.shuffle = shuffle

        # Image Transforms
        self.img_transforms = img_transforms  # A list of classes with a 'transform' function

        # Bathy Transforms
        self.raster_transforms = raster_transforms  # A list of classes with a 'transform' function

        # Set the optional image directory - images should have 'path' in them
        self.img_dir = img_dir

        if self.shuffle:
            # This shuffles in place
            random.shuffle(self.img_ids)

    def __len__(self):
        """
        Denotes the number of batches in an epoch

        Returns:
            (int): number of batches

        """
        return int(np.floor(len(self.img_ids)/self.batch_size))

    def on_epoch_end(self):
        """
        Operations to be performed on the end of each epoch. Must be called manually if not using .fit or .fit_generator
        Returns:

        """
        if self.shuffle:
            random.shuffle(self.img_ids)

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

            half_patch = np.floor(patch_size[0])

            off_x = int(np.round(px - half_patch))
            off_y = int(np.round(py - half_patch))

            patch = extract_raster_patch(self.raster_collection[raster]['dataset'], off_x, off_y, patch_size[0])
        else:
            patch = np.load(image[raster]['path'])

        if raster == "bathy" or raster == "bathymetry":  # Separate this, as the depth needs to be separated
            centre_depth = patch[round(patch.shape[0]/2), round(patch.shape[1]/2)]

            dmap = remove_depth_map_zeros(patch)
            dmap, mean_depth = make_depth_map_mean_zero(dmap)

            return dmap, mean_depth
        else:
            return patch


    def __getitem__(self, index):
        """
        Generates a batch of data
        Args:
            index:

        Returns:

        """
        # Gets the img_ids for this batch
        image_batch_ids = self.img_ids[index*self.batch_size:(index+1)*self.batch_size]

        # Get the images for this batch
        images = self.coco.loadImgs(image_batch_ids)

        # image/images: a dict/list of dicts containing image information according to coco standard
        # img/imgs: actual img (e.g. np array)

        batch_dict = {}
        batch_dict['images'] = []
        for r in self.rasters:
            if r == 'bathy' or r == 'bathymetry':
                batch_dict['dmap'] = []
                batch_dict['depth'] = []
            else:
                batch_dict[r] = []
        if self.use_categories:
            batch_dict['categories'] = []


        not_ok_count = 0

        # A while loop is here because it is possible that all data elements that are needed aren't present for each image
        while len(batch_dict['images']) < self.batch_size:
            # Get the next image index
            img_idx = self.img_ids[index]
            # Load the image entry
            image = self.coco.loadImgs([img_idx])[0]
            if self.img_dir:  # Not reccomended
                img_path = os.path.join(self.img_dir, image['file_name'])
            elif 'path' in image:
                img_path = image['path']
            else:
                # This will probably fail
                img_path = image['file_name']

            # Load the image
            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)

            # Load each raster and allocate them to the dictionary
            raster_patches = {}
            for r in self.rasters:
                raster_patches[r] = self.load_raster(raster=r, image=image)

            # Transforms the images
            if self.img_transforms:
                for trans in self.img_transforms:
                    img = trans.transform(img)

            # Transform each raster patch. Special case for bathymetry as depth is split off
            if self.raster_transforms:
                for r,patch in raster_patches.items():
                    if r == 'bathy' or r == 'bathymetry':
                        dmap = patch[0]  # In this case patch is a tuple (dmap, depth)
                        # Only operate on the depth map
                        for rt in self.raster_transforms[r]:
                            dmap = rt.transform(dmap)
                        # Add the patch tuple back in
                        raster_patches[r] = (dmap, patch[1])
                    else:
                        # Transform the patch
                        for rt in self.raster_transforms[r]:
                            patch = rt.transform(patch)
                        # Transform the patch
                        raster_patches[r] = patch

            # Extract the categories. For these datasets, the annotation is a single point label
            if self.use_categories:
                # Get annotations - should be just one per image
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
                if len(annotations) > 0:
                    cat = annotations[0]['category_id'] - 1  # Just get first annotation. Categories are indexed from 1.
                else:
                    cat = None
            else:
                cat = None

            # Check all data is collected properly
            # TODO how to handle NaNs ???
            # Check the image
            img_ok = True
            if img is None:
                img_ok = False
            # Check rasters
            rasters_ok = True  # This will be set False if any raster is not OK.
            for r,patch in raster_patches.items():
                if r == 'bathy' or r == 'bathymetry':
                    if np.isnan(patch[1]) or np.isnan(patch[0]).any():
                        dm = np.nan_to_num(patch[0], copy=True)
                        dep = np.nan_to_num(patch[1], copy=True)
                        raster_patches[r] = (dm, dep)
                        rasters_ok = True
                else:
                    if np.isnan(patch).any():
                        rasters_ok = False
            cat_ok = True
            # Check the categories
            if self.use_categories and cat is None:
                cat_ok = False

            if img_ok and rasters_ok and cat_ok:  # Check that all patches have been extracted
                batch_dict['images'].append(img)
                for r, patch in raster_patches.items():
                    if r == 'bathy' or r == 'bathymetry':
                        batch_dict['dmap'].append(patch[0])
                        batch_dict['depth'].append(patch[1])
                    else:
                        batch_dict[r].append(patch)
                if self.use_categories:
                    batch_dict['categories'].append(cat)
                not_ok_count = 0
            else:
                not_ok_count += 1
                if not_ok_count >= 10:
                    warnings.warn("Lots of data not ok")


            # Increment index, which in turn increments the image index
            index += 1
            if index >= len(self.img_ids):  # If img idx exceeds
                index = 0

        # Create a list of the returns
        return_list = []
        for k,v in batch_dict.items():
            if k == 'images':
                return_list.append(np.asarray(v))
            elif k != 'categories':  # Need to operate on the patches, in case the transforms have collapsed them (e.g. to a vector through a model transform)
                patch_array = np.asarray(v)
                if len(patch_array.shape) >= 3:  # If the patches are 3D (batch_size, width, height), make them 4D (batch_size, width, height, channels)
                    patch_array = np.expand_dims(np.asarray(v), axis=3)
                return_list.append(patch_array)
            elif k == 'categories':  # Want categories to be last
                return_list.append(keras.utils.to_categorical(v, num_classes=self.num_classes))
            else:
                raise ValueError("")

        return tuple(return_list)


    def print_output_format(self):
        """
        This generator returns different outputs depending on its configuration, e.g. number + type of rasters, whether categories are on. Calling this function prints the output format.
        Returns:
            None

        """
        batch_dict = {}
        batch_dict['images'] = []
        for r in self.rasters:
            if r == 'bathy' or r == 'bathymetry':
                batch_dict['dmap'] = []
                batch_dict['depth'] = []
            else:
                batch_dict[r] = []
        if self.use_categories:
            batch_dict['categories'] = []

        return_print_list = []

        for k, v in batch_dict.items():
            if k == 'images':
                return_print_list.append(k)
            elif k != 'categories':  # Need to operate on the patches, in case the transforms have collapsed them (e.g. to a vector through a model transform)
                return_print_list.append(k)
            elif k == 'categories':  # Want categories to be last
                return_print_list.append(k)
            else:
                raise ValueError("")
        print("Image output format is: ", ','.join(return_print_list))



