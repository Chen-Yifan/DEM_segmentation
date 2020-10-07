#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_50p_over \
--date 20building_aug \
--isTrain \
--threshold 1 \
--loss wbce \
--model unet \
--use_gradient \
--ckpt_name building_unet_wbce_gradient \
--num_filters 112 \
--input_channel 1 \
--n_epoch 90 \
--dim 128 \
--batch_size 16 \
--save_model \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
