#!/bin/bash
python main.py \
--dataroot /home/shared/dem/erosion_data/filtered_128_over \
--frame_name all_frames_5m6b/ \
--mask_name all_masks_5m6b/ \
--date aug \
--loss bce \
--ckpt_name unet32_DEM_wbce0.7_lr0.6_v2 \
--input_channel 1 \
--n_epoch 57 \
--dim 128 \
--batch_size 16 \
--visualize \