from PIL import Image
import numpy as np
import os
import re
import scipy.misc
import random
import sys
import csv
    
def is_feature_present(input_array):
    return (np.sum(input_array!=0)>10)     # select the image with more than 50 pixel label

    
def load_feature_data(dataroot, frame_dir, mask_dir, feature_type='erosion', dim=128):
    
    '''load frames and masks into two numpy array respectively
        -----
        condition: with feature
        arguments: 
            frame_dir, mask_dir, 
            feature_type: str, either erosion or building
            dim: width and height of the image
        
        process: always resize to 128x128 as model input
                normalize on local image maxx and minn
        -----
    '''
    low=0.1 
    hi=1.0
    test_frames = []
    test_masks = []
    test_masks_ext = []
    test_masks_MS = []
    frames = []
    masks = []
    name_list = []
    frame_names = os.listdir(frame_dir)
    frame_names.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])   # sort frame_names
                                    
    print("** load image from directory loop starts:")
    for i in range(len(frame_names)):
        frame_file = frame_names[i]
        # if len(frames)>1000:
            # break
            
        """find mapped frame and mask path"""
        frame_path = os.path.join(frame_dir, frame_file)

        """load image from tif and remove useless data"""
        
        if feature_type=='erosion':
            mask_path = os.path.join(mask_dir, frame_file)
            x = np.load(frame_path)
            # frame_array = np.concatenate((x[:,:,0:2], np.expand_dims(x[:,:,-1], axis=2)),axis=-1)
            frame_array = x[:,:,-1]
            label_array = np.load(mask_path)
        
        else: # building
            mask_file = frame_file.replace('mclean_fillnodata_','')
            mask_path = os.path.join(mask_dir, mask_file)
            #### for 128_0ver
            # mask_path = os.path.join(mask_dir, frame_file.replace('DEM','label'))
            if(frame_file[-3:]=='tif'):
                if not os.path.exists(mask_path):
                    print('rm mask_path', mask_path)
                    # os.remove(frame_path)
                    continue
                frame_array = np.array(Image.open(frame_path))
                label_array = np.array(Image.open(mask_path))    
            else:
                # os.remove(frame_path)
                # if os.path.exists(mask_path):
                    # os.remove(mask_path)
                    # print('remove1',frame_file)
                continue
                
            # check the dimension, if dimension wrong, remove
            dims = frame_array.shape
            if dims[0]!=dim or dims[1]!=dim or (len(np.unique(frame_array))<3): # remove the file if the frame has less than 3 unique data
                os.remove(mask_path)
                # os.remove(frame_path)
                print('remove2',frame_file)
                continue
            
        # both erosion and builiding, we check if feature is present
        if not is_feature_present(label_array):
            continue
        
        """Resize to dim"""
        if frame_array.shape[0]!=dim:
            frame_array = np.array(Image.fromarray(frame_array).resize((dim,dim), Image.BILINEAR))
            label_array = np.array(Image.fromarray(label_array).resize((dim,dim), Image.NEAREST))
        
        """Try preprocess : Normalization"""
        try:
            minn, maxx = np.min(frame_array[frame_array > 0]), np.max(frame_array[frame_array > 0])    
            frame_array[frame_array > 0] = low + (frame_array[frame_array > 0] - minn) * (hi - low) / (maxx - minn)
        except:
            continue
        
        # check label 0 1 2
        unique_labels = np.unique(label_array)
        label_array = np.where(label_array==2, 1, label_array)
        if 2 in unique_labels and 1 not in unique_labels:
            # load the manual labels
            manual_mask_path = os.path.join(dataroot, "label_manual_test/", mask_file)
            if not os.path.exists(manual_mask_path):
                continue
            test_frames.append(frame_array)
            # add the MS labels
            test_masks_MS.append(label_array)
            label_array = np.array(Image.open(manual_mask_path))
            test_masks_ext.append(label_array)
            label_array = np.where(label_array==2, 0, label_array) # only care the label 1
            test_masks.append(label_array)
            
        else:
            frames.append(frame_array)
            masks.append(label_array)
            
        name_list.append(frame_names[i])
            
    """Form array and name_list"""
    frames, masks, test_frames, test_masks, test_masks_ext, test_masks_MS = np.array(frames), np.array(masks), np.array(test_frames), np.array(test_masks), \
                                                                            np.array(test_masks_ext), np.array(test_masks_MS)
    print("meta data: training feature/bkground ratio",np.sum(masks),  np.sum(1-masks))
    
    """Extend to 4 dimensions for training """
    if(frames.ndim != 4):
        frames = np.expand_dims(frames, -1)
        test_frames = np.expand_dims(test_frames, -1)
    masks = np.expand_dims(masks, -1)
    test_masks = np.expand_dims(test_masks, -1)
    test_masks_ext = np.expand_dims(test_masks_ext, -1)
    test_masks_MS = np.expand_dims(test_masks_MS, -1)
    
    assert(test_masks.shape == test_masks_ext.shape)
    assert(test_masks.shape == test_masks_MS.shape)
    
    print("test_masks.shape = ", test_masks.shape)
    
    # split frames/masks to train:val = 5:1
    a = int(len(frames)*5/6)
    train_frames, train_masks = frames[:a], masks[:a]
    val_frames, val_masks = frames[a:], masks[a:]
    return train_frames, val_frames, test_frames, train_masks, val_masks, test_masks, test_masks_ext, test_masks_MS, name_list



def load_data(opt):
    """
    Load data to a dictionary containing train, val, test
    Return: Data_dict
    """
    
    train_frames, val_frames, test_frames, train_masks, val_masks, test_masks, test_masks_ext, test_masks_MS, name_list = \
                                load_feature_data(opt.dataroot, opt.frame_path, opt.mask_path, opt.dataset, opt.dim)
                                
    n_train, n_test, n_val = len(train_frames), len(test_frames), len(val_frames)
    print('***** #train: #test: #val = %d : %d :%d ******'%(n_train, n_test, n_val))

    Data_dict = {
            'train':[train_frames.astype('float32'),
                     train_masks.astype('float32')],
            'val':[val_frames.astype('float32'),
                    val_masks.astype('float32')],
            'test':[test_frames.astype('float32'),
                    test_masks.astype('float32')],
            'test_MS':[None,
                    test_masks_MS.astype('float32')],
            'test_ext':[None,
                    test_masks_ext.astype('float32')],
            }
    
    return Data_dict
