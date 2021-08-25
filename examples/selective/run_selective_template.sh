#!/bin/bash
MODE=SELECTED_MODE
EXPDIR=EXPERIMENT_DIR
DATASETDIR=DATASET_DIR
VALCOCO=VALIDATION_COCO

python3 ~/src/habitat-modelling/experiments/wae/bathymetry_to_label/selective_training_commander.py \
    --dataset-dir $DATASETDIR \
    --val-coco-path $VALCOCO \
    --scratch-dir $EXPDIR \
    --num-classes 5 \
    --encoders bathymetry \
    --encoder-params /media/hdd1/water/results/wae/20190703-wae-bae-basic-mid/encoder_params.json \
    --encoder-weights /media/hdd1/water/results/wae/20190703-wae-bae-basic-mid/models/b2b_bathy_encoder_5.pt \
    --rasters bathymetry \
    --latent-dim 33 \
    --layer-neurons 2048,2048,2048 \
    --model-type dropn2 \
    --batch-size 8 \
    --dropout 0.5 \
    --lr 0.0001 \
    --cuda \
    --epochs 10 \
    --training-iterations 15 \
    --initial-datasets INITIAL_DATASETS \
    --datasets-per-iteration 5 \
    --selection-mode $MODE
