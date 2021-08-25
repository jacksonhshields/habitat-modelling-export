import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import plyfile
import numpy as np
import utm
import math
import warnings
import random
from pycocotools.coco import COCO



class ArrayDataset(data.Dataset):
    def __init__(self, paths):
        """
        This dataset is useful for loading cached datasets.

        Initialises the generator

        Args:
            paths: (list) A list of paths to the .npy files containing the datasets
        """
        self.paths = paths

        self.array_list = [np.load(p) for p in paths]

        self.num_classes = None

    def __len__(self):
        """
        Denotes the number of images in each epoch

        Returns:
            (int): number of batches

        """
        return self.array_list[0].shape[0]

    def __getitem__(self, index):
        return [arr[index] for arr in self.array_list]


class CocoImage(data.Dataset):
    def __init__(self, coco_path, img_transform=None, use_categories=True, num_classes=None):
        from pycocotools.coco import COCO
        self.coco = COCO(coco_path)
        # Image Transforms
        self.img_transform = img_transform  # A torchvision transform
        self.img_ids = self.coco.getImgIds()

        self.use_categories = use_categories
        if self.use_categories:
            if num_classes is None:
                self.num_classes = len(self.coco.getCatIds())
            else:
                self.num_classes = num_classes

    def __len__(self):
        """
        Denotes the number of images in each epoch

        Returns:
            (int): number of batches

        """
        return len(self.img_ids)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, ?target). target is the object returned by ``coco.loadAnns``.
        """
        img_idx = index  # TODO store this?
        img_id = self.img_ids[img_idx]

        image = self.coco.loadImgs(ids=[img_id])[0]

        pimg = Image.open(image['path'])

        # img = np.array(pimg)

        # Extract the categories. For these datasets, the annotation is a single point label
        if self.use_categories:
            # Get annotations - should be just one per image
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image['id']]))
            if len(annotations) > 0:
                cat = annotations[0]['category_id']  # Just get first annotation. Categories are indexed.
                hot = np.zeros([self.num_classes])
                hot[cat] = 1.0
            else:
                cat = None
                hot = np.zeros([self.num_classes])
        else:
            cat = None

        if self.img_transform:
            img = self.img_transform(pimg)
        else:
            img = torch.tensor(np.array(pimg))
        timg = img

        return_list = []
        return_list.append(timg)

        if self.use_categories:
            return_list.append(torch.tensor(hot))

        return tuple(return_list)


