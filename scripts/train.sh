#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/data/selected_128_overlap \
--date erosion \
--isTrain 1 \
--ckpt_name 128over_unet_luo_sigmoid \
--input_channel 5 \
--n_epoch 40 \
--input_shape 128 \
--batch_size 16 \
--save_model 1 \
--frame_name all_frames_5m6b/ \
--mask_name all_masks_5m6b/ \
