#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_over/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--DEM_only \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name unet32_bce_DEM_bs32 \
--num_filters 32 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 32 \
--save_model

# Tverskey Focal loss
python main.py \
--dataroot /home/yifanc3/dataset/building_data/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--DEM_only \
--threshold 1 \
--loss T \
--model ourunet \
--ckpt_name ourunet112_T_DEM \
--num_filters 32 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

# lovasz loss
#!/bin/bash
python main.py \
--dataroot /home/yifanc3/dataset/building_data/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--DEM_only \
--threshold 1 \
--loss L \
--model unet \
--ckpt_name unet32_L_DEM \
--num_filters 32 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

## use wholemap
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--DEM_only \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name whole_unet112_bce_DEM \
--num_filters 112 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

# gradient
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--use_gradient \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name whole_unet32_bce_gradient \
--num_filters 32 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--DEM_only \
--threshold 1 \
--loss T \
--model unet \
--ckpt_name whole_unet32_T_DEM \
--num_filters 32 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

# aspect
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name whole_unet112_bce_aspect \
--num_filters 112 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

# slope
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name whole_unet112_bce_slope_v2 \
--num_filters 112 \
--input_channel 1 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model


# 4 channels  [array, slope, aspect, gradient]
python main.py \
--dataroot /home/yifanc3/dataset/building_data/128_0p_whole/ \
--frame_name DEM_retile/ \
--mask_name label_manual_mask/ \
--date 2105_building_manual \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name whole_unet32_bce_4c \
--num_filters 32 \
--input_channel 4 \
--n_epoch 150 \
--dim 128 \
--batch_size 16 \
--save_model

