#!/bin/bash
python main.py \
--dataroot /home/shared/dem/erosion_data/filtered_128_over \
--date aug \
--isTrain \
--loss wbce \
--input_channel 1 \
--augmentation \
--ckpt_name unet32_DEM_wbce0.7_lr0.6_v2 \
--num_filters 32 \
--model unet \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name all_frames_5m6b/ \
--mask_name all_masks_5m6b/ \
--save_model
