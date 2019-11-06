#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_0over \
--date 10.23 \
--ckpt_name building_DEM_dgen_Unet_128 \
--n_epoch 40 \
--input_shape 128 \
--batch_size 16 \
--save_model 1 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
