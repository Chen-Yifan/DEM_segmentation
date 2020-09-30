#!/bin/bash
python main.py \
--dataroot /home/shared/dem/erosion_data/filtered_128_over \
--date aug \
--isTrain \
--loss bce \
--threshold 1 \
--input_channel 1 \
--ckpt_name unet32_hillshade \
--num_filters 32 \
--model unet \
--n_epoch 300 \
--dim 128 \
--batch_size 16 \
--frame_name all_frames_5m6b/ \
--mask_name all_masks_5m6b/ \
--save_model \
