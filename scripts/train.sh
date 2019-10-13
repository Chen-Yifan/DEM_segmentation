#!/bin/bash
python cv_main.py \
--date 10.1nonorm \
--name 128over5m7bfill_DEM_unet_Adadelta_200w_30e \
--model Unet \
--loss_function single \
--epoch 30 \
--augmentation 0 \
--mask_name all_masks_5m6b \
--weight 200 \
--frame_name all_frames_5m7b_fill \
--optimizer 1 \
--k 2 \
--input_channel 7 \
--derivative 0 \
