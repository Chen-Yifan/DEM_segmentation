#!/bin/bash
python main.py \
--date 21building_cln \
--threshold 1 \
--loss bce \
--ckpt_name unet32_T_DEM_v2_part3_ispresent \
--n_epoch 120 \
--dim 128 \
--batch_size 16 