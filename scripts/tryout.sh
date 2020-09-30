#!/bin/bash
python cv_main.py \
--date 10.9pretrain \
--name 128over5m6b_Unet_half_Adadelta_der3_200w_30e \
--model Unet \
--loss_function single \
--epoch 30 \
--augmentation 0 \
--mask_name all_masks_5m6b \
--weight 200 \
--frame_name all_frames_5m6b \
--optimizer 1 \
--k 2 \
--derivative 6 \
--load_pretrained 1 \
--weights_path /home/yifanc3/tumor_segmentation_model.h5 \
