#!/usr/bin/env bash

set -x

python -u main.py \
    --output_dir "exps/swin_ddetr_fixed" \
    --backbone "swinfixed" \
    --batch_size 2