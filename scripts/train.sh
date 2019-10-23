#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/hackathon_dataset \
--date 10.23 \
--ckpt_name experiment \
--n_epoch 30 \
--input_shape 128 \
--batch_size 16 \
--save_model 1 \
--frame_name all_frames_DEM/ \
--mask_name all_masks_5m6b/ \
