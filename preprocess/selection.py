import os
import random
import re
from PIL import Image
from libtiff import TIFF
import numpy as np
from shutil import copyfile

DATA_PATH = '/home/yifanc3/dataset/data/'
FRAME_PATH = DATA_PATH+'frames_64/'
MASK_PATH = DATA_PATH+'masks_64/'
ANNO_PATH = DATA_PATH+'annotations_64/'
SAVE_PATH = '/home/yifanc3/dataset/data/selected/'


#annotated_mask
# this is wrong , we need to split annotation image to 64*64 too.
# annotated_mask = np.load(os.path.join(DATA_PATH,'annotated_mask.tif')) # only 0 has annotated 
                         
# Create folders to hold features and nonfeatures

folders = ['frame_0', 'mask_0', 'frame_1', 'mask_1']

# os.makedirs(SAVE_PATH)
# for folder in folders:
#     os.makedirs(SAVE_PATH + folder)

  
# Get all frames and masks, sort them, shuffle them to generate data sets.

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

Ratios = []
num_files = 0
num_anno = 0
num_feature_file = 0
#select files with feature
for mask in all_masks:
    num_files += 1
    # load annotations
    filename = os.path.join(ANNO_PATH, mask)
    tif = TIFF.open(filename)
    img = tif.read_image().astype(int) # 0 means annotated
    if 0 not in np.unique(img):
        continue
    print(mask)
    num_anno += 1
    #load mask image
    filename = os.path.join(MASK_PATH, mask)
    tif = TIFF.open(filename)
    mask_arr = tif.read_image().astype(int)
    mask_arr = np.where(mask_arr==2.0, 0.0, mask_arr)
    mask_arr = 1 - mask_arr # so 1 is the mask
    # calculate ratio
    no_1 = np.count_nonzero(mask_arr)
    ratio = no_1/(64*64)
    if(no_1 != 0):
        Ratios.append(ratio)
    if(no_1 >= 10):
        num_feature_file += 1
        # load frames
        filename = os.path.join(FRAME_PATH, mask)
        tif = TIFF.open(filename)
        frame_arr = tif.read_image()
        # save to file masks_1 and load frames
        np.save(os.path.join(SAVE_PATH+'frame_1/',mask[0:-4]+'.npy'), frame_arr)
        np.save(os.path.join(SAVE_PATH+'mask_1/',mask[0:-4]+'.npy'),mask_arr)
        
        

#read all the masks to see if 1/0 >= 10/(64*64)
# the ratio need to be calculated by histgram (plot histogram of how large ratio is the label/nolabel )
                         
Ratios = []
num_files = 0
num_anno = 0
num_feature_file = 0
num_nonfeature_file = 0
#select files with feature
random_list = np.random.randint(2, size=81703)

# for non features
for mask in all_masks:
    num_files += 1
    # load annotations
    filename = os.path.join(ANNO_PATH, mask)
    tif = TIFF.open(filename)
    img = tif.read_image().astype(int) # 0 means annotated
    if 0 not in np.unique(img):
        if(random_list[num_files] == 1 && num_nonfeature_file < 1195):
            # save this frame and mask
            num_nonfeature_file += 1 # count
            
            filename = os.path.join(MASK_PATH, mask)
            tif = TIFF.open(filename)
            mask_arr = tif.read_image().astype(int)
            mask_arr = np.where(mask_arr==2.0, 0.0, mask_arr)
            mask_arr = 1 - mask_arr # so 1 is the mask
            
            filename = os.path.join(FRAME_PATH, mask)
            tif = TIFF.open(filename)
            frame_arr = tif.read_image()
            # save to file masks_1 and load frames
            np.save(os.path.join(SAVE_PATH+'frame_0/',mask[0:-4]+'.npy'), frame_arr)
            np.save(os.path.join(SAVE_PATH+'mask_0/',mask[0:-4]+'.npy'),mask_arr)              
        continue
        