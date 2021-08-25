#!/usr/bin/env python3


#!/usr/bin/env python3
import argparse
import os
from pycocotools.coco import COCO
from habitat_modelling.ml.torch.transforms.image_transforms import create_image_transform_list_from_args

def get_args():
    parser = argparse.ArgumentParser(description="Splits a raster by class")
    parser.add_argument("coco", type=str, help="The input raster to split")
    parser.add_argument("--transforms", type=str, help="The transforms to perform on the dataset")
    args = parser.parse_args()
    return args

def main(args):
    coco = COCO(args.coco)



    for image in coco.loadImgs(coco.getImgIds()):
        pimg =




if __name__ == "__main__":
    main(get_args())