#!/bin/bash
python main.py \
--date 20building_clip \
--threshold 1 \
--loss bce \
--ckpt_name unet64_bce_DEM_v2_whole \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
