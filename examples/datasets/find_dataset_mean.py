#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os
import json
from pycocotools.coco import COCO
import copy


def get_args():
    parser = argparse.ArgumentParser(description="Creates a coco json from an ACFR Marine Dive")
    parser.add_argument('input_coco', help="Path to the coco format dataset")
    parser.add_argument('--output_coco', help="Path to the output coco file. If given will add mean to info")
    parser.add_argument('--image_dir', help="Optional image directory. Not reccomended, uses paths instead")
    args = parser.parse_args()
    return args


def main(args):
    coco = COCO(args.input_coco)


    count = 0

    cumR = 0.
    cumG = 0.
    cumB = 0.

    sumR = 0.
    sumG = 0.
    sumB = 0.


    for image in coco.loadImgs(coco.getImgIds()):
        if args.image_dir:
            img_path = os.path.join(args.image_dir, image['file_name'])
        elif 'path' in image:
            img_path = image['path']
        else:
            img_path = image['file_name']

        img = cv2.imread(img_path, 1)  # Read colour image
        if img is not None:
            (B, G, R) = cv2.split(img.astype("float32"))

            rMean = np.mean(R)
            gMean = np.mean(G)
            bMean = np.mean(B)

            cumR = cumR + (rMean - cumR)/(count + 1)
            cumG = cumG + (gMean - cumG)/(count + 1)
            cumB = cumB + (bMean - cumB)/(count + 1)

            sumR += rMean
            sumG += gMean
            sumB += bMean

            count += 1

    print("Moving: Red %f, Green %f, Blue %f" %(cumR, cumG, cumB))
    print("Static: Red %f, Green %f, Blue %f" %(sumR/float(count), sumG/float(count), sumB/float(count)))

    if args.output_coco:
        output = copy.deepcopy(coco.dataset)
        output['info']['channel_means'] = {"red", cumR, "green", cumG, "blue", cumB}
        json.dump(output, open(args.output_coco, 'w'))

if __name__ == "__main__":
    main(get_args())
