#!/usr/bin/env bash

DIVE_PATH="/media/hdd1/water/processed/2008/Tasmania200810/r20081014_052323_ohara_20_oneline"
RENAV_SUBDIR="renav20160205"
OUTPUT_DIR="/media/hdd1/water/datasets/2008/Tasmania200810/r20081014_052323_ohara_20_oneline/"
OUTPUT_COCO="$OUTPUT_DIR/ohara_20_bathy.json"
RENAV_POSE_FILE="/media/hdd1/water/processed/2008/Tasmania200810/r20081014_052323_ohara_20_oneline/renav20160205/stereo_pose_est.data"
BATHYMETRY_TIF="/media/hdd1/water/Tassie_bathy/Tasmania200810/TAFI_provided_data/BathymetryAsTiffs/fort1.tif"
BATHYMETRY_PATCH_DIR="$OUTPUT_DIR/bathy_patches/"
TRIP="Tasmania200810"
YEAR="2008"

~/src/habitat-modelling/applications/dive-to-coco \
    $DIVE_PATH \
    $OUTPUT_COCO \
    --renav-pose-file $RENAV_POSE_FILE \
    --bathymetry $BATHYMETRY_TIF \
    --extract-bathymetry-patches \
    --bathy-patch-size 21,21  \
    --bathymetry-patch-dir $BATHYMETRY_PATCH_DIR \
    --trip $TRIP \
    --year $YEAR
