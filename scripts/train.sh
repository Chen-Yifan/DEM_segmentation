#!/bin/bash
python cv_main.py \
--date tryout \
--name 128over5m6b_unet_Adadelta_200w_100e \
--model unet \
--loss_function single \
--epoch 1 \
--augmentation 1 \
--mask_name all_masks_5m6b \
--weight 1.0 \
--frame_name all_frames_5m6b_norm \
--optimizer 1 \
--k 2