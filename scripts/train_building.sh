#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss wbce \
--model unet \
--ckpt_name unet64_wbce_DEM \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile2/ \
--save_model

python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss wbce \
--model unet \
--ckpt_name unet64_wbce_DEM_v2 \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile2/ \
--save_model


python main.py \
--dataroot /home/shared/dem/building_data/128_50p_over \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 0 \
--loss L \
--model unet \
--ckpt_name unet64_L_DEM_v2_v2 \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
--save_model


# test
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name unet64_bce_DEM_v2 \
--num_filters 64 \
--input_channel 1 \
--n_epoch 80 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile2/


python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet_rgl \
--ckpt_name unetrgl64_bce_DEM \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile \
--save_model 

