#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/building_data/512_50p_over \
--date 10.23 \
--ckpt_name building_DEM_dgen \
--n_epoch 20 \
--input_shape 512 \
--batch_size 16 \
--save_model 1 \
--frame_name labels_retile/ \
--mask_name DEM_retile/ \
