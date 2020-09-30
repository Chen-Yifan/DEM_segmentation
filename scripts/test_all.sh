#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_50p_over \
--date 20building_aug \
--threshold 1 \
--loss bce \
--ckpt_name 128over_unet112_bce_DEM_clip0over \
--n_epoch 33 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
