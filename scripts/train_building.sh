#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_50p_over \
--date 20building_aug \
--isTrain \
--threshold 1 \
--loss bce \
--model resnet \
--use_gradient \
--ckpt_name building_resnet34_bce_gradient \
--num_filters 112 \
--input_channel 1 \
--n_epoch 60 \
--dim 128 \
--batch_size 16 \
--save_model \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
