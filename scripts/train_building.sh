#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 21building_cln \
--dataset building \
--isTrain \
--threshold 1 \
--loss T \
--model unet \
--ckpt_name unet32_T_DEM_part1_ispresent \
--num_filters 32 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name label_cln/ \
--save_model

# v2 100 Lovasz loss
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 21building_clip_100v2 \
--dataset building \
--isTrain \
--threshold 1 \
--loss L \
--model unet \
--ckpt_name unet32_lovasz_DEM \
--num_filters 32 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name label_100_v2/ \
--save_model


# v2 100 wbce
#!/bin/bash
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 21building_clip_100v2 \
--dataset building \
--isTrain \
--threshold 1 \
--loss wbce \
--model unet \
--ckpt_name unet64_wbce_DEM \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name label_100_v2/ \
--save_model


# continue train
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 21building_clip_100v2 \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet_rgl \
--ckpt_name unetrgl64_bce_DEM \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name label_100_v2/ \
--save_model

# origin v1
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name unet64_bce_DEM_part2_he \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
--save_model


python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet_rgl \
--ckpt_name unetrgl64_DEM_v3 \
--num_filters 64 \
--pretrained_weights ./checkpoints/20building_clip/unetrgl64_DEM_v3/weights.14-0.6301-0.0000.hdf5 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile/ \
--save_model


# test
python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--threshold 1 \
--loss bce \
--model unet \
--ckpt_name unet64_bce_DEM_v2 \
--num_filters 64 \
--input_channel 1 \
--n_epoch 80 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile2/


python main.py \
--dataroot /home/shared/dem/building_data/128_0over_clip \
--date 20building_clip \
--dataset building \
--isTrain \
--threshold 1 \
--loss bce \
--model unet_rgl \
--ckpt_name unetrgl64_bce_DEM \
--num_filters 64 \
--input_channel 1 \
--n_epoch 100 \
--dim 128 \
--batch_size 16 \
--frame_name DEM_retile/ \
--mask_name labels_retile \
--save_model 

