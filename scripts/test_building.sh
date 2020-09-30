#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_50p_over \
--date 20building_aug \
--threshold 1 \
--loss bce \
--ckpt_name unet16_bce_DEM \
--n_epoch 68 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
