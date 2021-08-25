#!/usr/bin/env bash

ITERATIONDIR=$1
VALCOCO=$2

ENCODER_PARAMS=/media/hdd1/water/results/wae/20190703-wae-bae-basic-mid/encoder_params.json
ENCODER_WEIGHTS=/media/hdd1/water/results/wae/20190703-wae-bae-basic-mid/models/b2b_bathy_encoder_5.pt

mkdir -p $ITERATIONDIR/validation/cache

for idir in $( find $ITERATIONDIR -name "iteration_*" -maxdepth 1); do
    echo $idir
    cd $idir
    model_params=$idir/models/model_params.json
    model_weights=$idir/models/latent_model_best.pt
    python3 ~/src/habitat-modelling/experiments/wae/bathymetry_to_label/validate_neighbours.py \
        $VALCOCO \
        --cached-dataset-dir $ITERATIONDIR/validation/cache \
        --params $idir/params.json \
        --classifier-params $model_params \
        --classifier-weights-path $model_weights \
        --model-type dropn2 \
        --scratch-dir validation \
        --encoders bathymetry \
        --encoder-params $ENCODER_PARAMS \
        --encoder-weights $ENCODER_WEIGHTS \
        --loss kl \
        --num-classes 5 \
        --rasters bathymetry \
        --raster-sizes 21x21 \
        --inference-samples 16 \
        --output-activation log_softmax \
        --cuda
    cd $ITERATIONDIR
done

