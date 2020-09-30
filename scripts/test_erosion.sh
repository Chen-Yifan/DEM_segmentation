#!/bin/bash
python main.py \
--dataroot /home/shared/dem/erosion_data/filtered_128_over \
--date flip_aug \
--threshold 1 \
--loss bce \
--ckpt_name unet_bce_DEM_16_v2 \
--input_channel 1 \
--n_epoch 111 \
--dim 128 \
--batch_size 16 \
--frame_name all_frames_5m6b/ \
--mask_name all_masks_5m6b/ \
