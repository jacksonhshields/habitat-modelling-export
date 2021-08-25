#!/bin/bash

datasetdir=datasets
valcoco=val/ohara20.balanced.json
splits=$( $1 | sed 's#,#\ #g' )
splitsfilt=$( for sp in $splits; do if [ -f "$datasetdir/part${sp}.json" ]; then echo $sp; fi;  done )

splitdir=$( echo $splitsfilt | sed 's#\ #_#g' )
comsplits=$( echo $splitsfilt | sed 's#\ #,#g' )

initial_json=$(for sp in $splitsfilt; do echo part${sp}.json ; done)
initials=$( echo $initial_json | sed 's#\ #,#g' )

modes="epistemic_mean aleatoric_mean random"

# Setup datasets

for mod in $modes; do
    mkdir -p $(pwd)/$splitdir/$mod
    cat run_selective_template.sh | \
        sed "s#SELECTED_MODE#$mod#g" | \
        sed "s#EXPERIMENT_DIR#$(pwd)/$splitdir/$mod#g" | \
        sed "s#INITIAL_DATASETS#$initials#g" | \
        sed "s#DATASET_DIR#$datasetdir#g" | \
        sed "s#VALIDATION_COCO#$valcoco#g" > $(pwd)/$splitdir/$mod/run_selective.sh
    chmod +x $(pwd)/$splitdir/$mod/run_selective.sh
done


for mod in $modes; do
   ( cd $(pwd)/$splitdir/$mod && bash run_selective.sh )
done

