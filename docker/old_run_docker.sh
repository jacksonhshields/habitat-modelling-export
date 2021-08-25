#!/bin/bash


# PARAMS
ACTIVE_DATASET_ROOT="/mnt/hdd1/active-vision-dataset/ActiveVisionDataset/"
RETINA_ANNOTATIONS_DIR="/mnt/hdd1/active-vision-dataset/splits/set2/retina_annotations"
COCO_WEIGHTS="/mnt/hdd1/coco-weights/resnet50_coco_best_v2.1.0.h5"

xhost +local:root


nvidia-docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $ACTIVE_DATASET_ROOT:/ActiveVisionDataset \
    -v $RETINA_ANNOTATIONS_DIR:/retina-annotations \
    -v $COCO_WEIGHTS:/retinanet_resnet_coco_weights.h5 \
    retina \
    /bin/bash -c "ln -s /ActiveVisionDataset /retina-annotations/; bash"
