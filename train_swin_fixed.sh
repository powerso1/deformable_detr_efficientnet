#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/swin_ddetr_fixed" \
    --backbone "swinfixed" \
    --batch_size 2