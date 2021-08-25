#!/usr/bin/env bash


ITERATIONDIR=$1

# Usage:
# bash ~/src/habitat-modelling/examples/selective/clean_up_models.sh $(pwd)/random


for modeldir in $( find $ITERATIONDIR -name models ); do
    cd $modeldir
    nummods=$( ls | wc -l)

    fulldstlink=$(find latent_model_best.pt -maxdepth 0 -printf %l)
    bestmodel=$( basename $fulldstlink )
        unlink latent_model_best.pt
    if (( nummods > 4 )); then
        mkdir -p tmp
        mv *.pt tmp
        mv tmp/$bestmodel .
        rm -r tmp
        ln -s $bestmodel latent_model_best.pt
    else
        ln -s $bestmodel latent_model_best.pt
    fi
done