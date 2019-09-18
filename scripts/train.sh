#!/bin/bash
python cv_main.py \
--date tryout \
--name 128over5m6b_deeplab_Adadelta_200w_100e \
--model deeplab \
--loss_function single \
--epoch 100 \
--augmentation 1 \
--mask_name all_masks_5m6b \
--weight 1.0 \
--frame_name all_frames_5m6b_norm \
--optimizer 1 \
--k 2