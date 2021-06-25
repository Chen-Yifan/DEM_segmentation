#!/bin/bash
python main.py \
--date 2105_building_manual \
--threshold 1 \
--loss bce \
--ckpt_name ourunet112_T_DEM \
--DEM_only \
--n_epoch 22 \
--dim 128 \
--batch_size 16 

python main.py \
--date 2105_building_manual \
--threshold 1 \
--loss bce \
--ckpt_name whole_unet112_bce_DEM/ \
--n_epoch 54 \
--DEM_only \
--dim 128 \
--batch_size 16

# DEM
python main.py \
--date 2105_building_manual \
--threshold 1 \
--loss T \
--DEM_only \
--ckpt_name whole_unet32_T_DEM/ \
--n_epoch 74 \
--dim 128 \
--batch_size 16

# aspect
python main.py \
--date 2105_building_manual \
--threshold 1 \
--loss bce \
--ckpt_name whole_unet112_bce_slope_v2/ \
--n_epoch 85 \
--dim 128 \
--batch_size 16