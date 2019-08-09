import os
import random
import re
from PIL import Image
from libtiff import TIFF
import numpy as np
from shutil import copyfile

DATA_PATH = '/home/yifanc3/dataset/data/'
FRAME_PATH = DATA_PATH+'frames_256_overlap6/'
MASK_PATH = DATA_PATH+'masks_10m_256overlap/'
MASK2_PATH = DATA_PATH+'masks_5m_256overlap/'
ANNO_PATH = DATA_PATH+'annotations_256_overlap/'
SAVE_PATH = '/home/yifanc3/dataset/data/selected_256_overlap/'


#annotated_mask
# this is wrong , we need to split annotation image to 64*64 too.
# annotated_mask = np.load(os.path.join(DATA_PATH,'annotated_mask.tif')) # only 0 has annotated 
                         
# Create folders to hold features and nonfeatures

folders = ['frame_0', 'mask_5m_0', 'frame_1', 'mask_5m_1', 'mask_10m_1', 'mask_10m_0']

# os.makedirs(SAVE_PATH)

for folder in folders:
    os.makedirs(SAVE_PATH + folder)

  
# Get all frames and masks, sort them, shuffle them to generate data sets.

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)


num_feature_file = 1705
num_anno = 7365
num_feature_nonzero = 1804
#read all the masks to see if 1/0 >= 10/(64*64)
# the ratio need to be calculated by histgram (plot histogram of how large ratio is the label/nolabel )

random_list = np.concatenate((np.ones(num_feature_file),np.zeros(num_anno-num_feature_nonzero-num_feature_file))) #print(a,b,'\t',darray) 
np.random.shuffle(random_list)

num_files = 0
num_anno = 0
num_no_anno = 0 
# add to num_files
num_feature_nonzero = 0 
num_feature_zero = 0 #count index for random_list
# add to num_anno
num_feature_file = 0 
num_nofeature_file = 0 # these two number should be the same

#select files with feature and without feature
for mask in all_masks:
    num_files += 1
    # load annotations
    filename = os.path.join(ANNO_PATH, mask)
    tif = TIFF.open(filename)
    anno_img = tif.read_image().astype(int) # 0 means annotated
    if 0 not in np.unique(anno_img):
        num_no_anno += 1
        continue
    # annoated 
    print('annoted:',mask)
    num_anno += 1
    #load mask image
    filename = os.path.join(MASK_PATH, mask)
    tif = TIFF.open(filename)
    mask_arr = tif.read_image().astype(int)
#     mask_arr = np.where(mask_arr==2.0, 0.0, mask_arr)
#     mask_arr = 1 - mask_arr # so 1 is the mask
    # calculate ratio
    no_1 = np.count_nonzero(mask_arr)
    print(no_1)
            
    if(no_1 != 0):
        num_feature_nonzero += 1
    else: # no feature but annotated 
        if random_list[num_feature_zero] != 1:
            num_feature_zero += 1 
            continue
        filename = os.path.join(FRAME_PATH, mask)
        tif = TIFF.open(filename)
        frame_arr = tif.read_image()
        # save to file masks_1 and load frames
        print('nonfeature file')
        np.save(os.path.join(SAVE_PATH+'frame_0/',mask[0:-4]+'.npy'), frame_arr)
        np.save(os.path.join(SAVE_PATH+'mask_10m_0/',mask[0:-4]+'.npy'),mask_arr)
        
        num_feature_zero += 1 
        num_nofeature_file += 1
    if(no_1 >= 100):
        print('feature file')
        num_feature_file += 1
        # load frames
        filename = os.path.join(FRAME_PATH, mask)
        tif = TIFF.open(filename)
        frame_arr = tif.read_image()
        # save to file masks_1 and load frames
        np.save(os.path.join(SAVE_PATH+'frame_1/',mask[0:-4]+'.npy'), frame_arr)
        np.save(os.path.join(SAVE_PATH+'mask_10m_1/',mask[0:-4]+'.npy'),mask_arr)
    
print(num_files,num_anno, num_no_anno, num_feature_nonzero, num_feature_zero, num_feature_file, num_nofeature_file)
        