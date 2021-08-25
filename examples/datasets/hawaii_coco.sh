#!/usr/bin/env bash

DIVE_PATH="/media/water/processed/2018/Hawaii201801/r20180202_195659_SS11_waikoloa_broad_cros_200_150/"
OUTPUT_DIR="/media/water/datasets/2018/Hawaii201801/r20180202_195659_SS11_waikoloa_broad_cros_200_150/"
OUTPUT_COCO="$OUTPUT_DIR/waikoloa_broad_cross_200_150.json"
RENAV_SUBDIR="renav20180204/"
RENAV_POSE_FILE="$DIVE_PATH/$RENAV_SUBDIR/stereo_pose_est.data"
BATHYMETRY_TIF="/media/hdd1/water/thirdparty-data/FK180119_BigIsland/FK180119_Hawaii_3m_WGS84_CUBE_thru_4Feb_floatingPoint.tif"
BATHYMETRY_PATCH_DIR="$OUTPUT_DIR/bathy_patches/"
TRIP="Hawaii201801"
YEAR="2018"

BATHYMETRY_TIFF=/media/hdd1/water/thirdparty-data/FK180119_BigIsland/FK180119_Hawaii_3m_WGS84_CUBE_thru_4Feb_floatingPoint.tif
BACKSCATTER_TIFF=/media/hdd1/water/thirdparty-data/FK180119_BigIsland/FK180119_Hawaii_Backscatter_WGS84_2m_Feb4.tif

~/src/habitat-modelling/applications/dive-to-coco \
    $DIVE_PATH \
    $OUTPUT_COCO \
    --renav-pose-file $RENAV_POSE_FILE \
    --raster bathy,backscatter \
    --raster-tiffs $BATHYMETRY_TIF,$BACKSCATTER_TIFF \
    --extract-raster-patches \
    --raster-patch-size 21x21,21x21 \
    --raster-patch-dir $OUTPUT_DIR/patches/ \
    --trip $TRIP \
    --year $YEAR
