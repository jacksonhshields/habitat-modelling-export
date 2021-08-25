#!/usr/bin/env python3

import argparse
import json
from pycocotools.coco import COCO
import pymap3d
import numpy as np
import copy

def get_args():
    parser = argparse.ArgumentParser(description="Thins out a coco dataset based on the distance between images")
    parser.add_argument('coco_path', help="Path to the coco path dataset")
    parser.add_argument('output_coco', help="Path to the output coco dataset")
    parser.add_argument('--distance', '-d', type=float, required=True, help="The minimum distance between sequential images")
    args = parser.parse_args()
    return args

def args_to_cfg(args):
    cfg = {
        "coco_path": args.coco_path,
        "output_coco": args.output_coco,
        "distance": args.distance
    }
    return cfg


def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))

def main(cfg):
    coco = COCO(cfg['coco_path'])
    image_list = []
    ann_list = []


    last_pos_local = None
    datum = None
    for image in coco.loadImgs(coco.getImgIds()):
        add_image = False
        if 'geo_location' not in image:
            continue
        llh = image['geo_location']
        if datum is None:
            datum = llh  # Set a datum

        enu = pymap3d.geodetic2enu(lat=llh[0],lon=llh[1],h=0.,lat0=datum[0],lon0=datum[1], h0=0.)
        pos = np.array([enu[0], enu[1]])
        if last_pos_local is None:
            add_image = True
            last_pos_local = pos
        else:
            if dist(pos, last_pos_local) > cfg['distance']:
                add_image = True
                last_pos_local = pos

        if add_image:
            image_list.append(image)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))
            if len(anns) > 0:
                ann_list.extend(anns)

    dataset = copy.deepcopy(coco.dataset)
    dataset['images'] = image_list
    dataset['annotations'] = ann_list
    json.dump(dataset, open(cfg['output_coco'], 'w'), indent=4)


if __name__ == "__main__":
    main(args_to_cfg(get_args()))