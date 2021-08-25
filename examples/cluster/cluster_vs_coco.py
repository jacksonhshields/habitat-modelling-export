#!/usr/bin/env python3
import argparse
import json
from pycocotools.coco import COCO
import pandas as pd
from habitat_modelling.utils.raster_utils import extract_geotransform, get_pixels_from_lla, extract_raster_patch

def get_args():
    parser = argparse.ArgumentParser(description="Displays cluster assignment vs coco categories")
    parser.add_argument('--coco', help="Path to the coco file", required=True)
    parser.add_argument('--category-raster', help="path to the category raster (e.g. output from inference)")
    parser.add_argument('--save-dir', help="Directory to save confusion matrices")
    parser.add_argument('--show', action="store_true", help="Whether to show the plots")
    args = parser.parse_args()
    return args

def main(args):
    coco = COCO(args.coco)

    raster_ds, geotransform, raster_info = extract_geotransform(args.category_raster)

    category_to_cluster = {c:[] for c in coco.getCatIds()}

    for image in coco.loadImgs(coco.getImgIds()):
        px, py = get_pixels_from_lla(image['geo_location'], geotransform, raster_info['projection'])
        cluster = extract_raster_patch(raster_ds, int(px), int(py), 1)
        if cluster is None or cluster < 0:
            continue
        cluster = int(cluster)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image['id']]))

        if len(anns) == 0:
            continue

        cat = anns[0]['category_id']

        category_to_cluster[cat].append(cluster)
    # category_cluster_df = pd.DataFrame.from_dict(category_to_cluster)
    # print(category_cluster_df)
    print(category_to_cluster)

if __name__ == "__main__":
    main(get_args())


