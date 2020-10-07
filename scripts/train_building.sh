#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_0over \
--date 20building_aug \
--isTrain \
--threshold 1 \
--loss wbce \
--model unet \
--ckpt_name unet64_wbce7_lr6_clip0over \
--lr 0.001 \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
--save_model 