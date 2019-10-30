#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/data/selected_128_overlap \
--date 10.31 \
--ckpt_name trySDG \
--n_epoch 20 \
--input_shape 128 \
--batch_size 16 \
--save_model 1 \
--frame_name all_frames_DEM/ \
--mask_name all_masks_10m6b/ \
