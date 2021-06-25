from PIL import Image
import numpy as np
import os
import re
import scipy.misc
import random
import sys
import csv

# part 1/2
# def train_test_val_split(frame_data,mask_data, name_list, result_path):
#     n = len(frame_data)
#     a = int(n*0.75)
#     b = int(n*0.85)
#     # part2
#     frame_data = np.flip(frame_data,0)
#     mask_data = np.flip(mask_data,0)
    
#     # record test_names list in a csv
#     test_names = name_list[b:]
#     print('len test_files', len(test_names))
#     with open(result_path+'/test_names.csv', 'w') as myfile:
#         wr = csv.writer(myfile, dialect='excel')
#         wr.writerow(test_names)
#     x_train,x_val,x_test = frame_data[:a],frame_data[a:b],frame_data[b:]
#     y_train,y_val,y_test = mask_data[:a],mask_data[a:b],mask_data[b:]
#     return x_train, x_val, x_test, y_train, y_val, y_test
    
# part 3
def train_test_val_split(frame_data,mask_data, name_list, result_path):
    n = len(frame_data)
    a = int(n*0.75)
    b = int(n*0.90)
    frame_data = np.flip(frame_data,0)
    mask_data = np.flip(mask_data,0)
    
    # record test_names list in a csv
    test_names = name_list[b:]
    print('len test_files', len(test_names))
    with open(result_path+'/test_names.csv', 'w') as myfile:
        wr = csv.writer(myfile, dialect='excel')
        wr.writerow(test_names)
    x_train,x_test, x_val = frame_data[:a],frame_data[a:b],frame_data[b:]
    y_train,y_test, y_val = mask_data[:a],mask_data[a:b],mask_data[b:]
    return x_train, x_val, x_test, y_train, y_val, y_test
    
    
def is_feature_present(input_array):
    return (np.sum(input_array)>10)     # select the image with more than 50 pixel label

    
def load_feature_data(frame_dir, mask_dir, feature_type='erosion', dim=128):
    
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
    frames = []
    masks = []
    name_list = []
    frame_names = os.listdir(frame_dir)
    frame_names.sort(key=lambda var:[int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])    # sort frame_names
                                    
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
        
        else:
            mask_path = os.path.join(mask_dir, frame_file.replace('fillnodata','building_label'))
            #### for 128_0ver
            # mask_path = os.path.join(mask_dir, frame_file.replace('DEM','label'))
            if(frame_file[-3:]=='tif'):
                if not os.path.exists(mask_path):
                    # os.remove(frame_path)
                    continue
                frame_array = np.array(Image.open(frame_path))
                label_array = np.array(Image.open(mask_path))    
            else:
                # os.remove(frame_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    print('remove1',frame_file)
                continue
                
            # check the dimension, if dimension wrong, remove
            dims = frame_array.shape
            if dims[0]!=dim or dims[1]!=dim or (len(np.unique(frame_array))<3): # remove the file if the frame has less than 3 unique data
                # os.remove(mask_path)
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
        
        name_list.append(frame_names[i])
        frames.append(frame_array)
        masks.append(label_array)
            
    """Form array and name_list"""
    print(len(frames), len(masks))
    frames, masks = np.array(frames), np.array(masks)
    
    """Extend to 4 dimensions for training """
    if(frames.ndim != 4):
        frames = frames.reshape((len(frames), dim, dim, 1))
    masks = masks.reshape((len(masks),dim, dim, 1))
    
    return frames, masks, name_list



def load_data(opt):
    """
    Load data to a dictionary containing train, val, test
    Return: Data_dict
    """
    frame_data, mask_data, name_list = load_feature_data(opt.frame_path, opt.mask_path, opt.dataset, opt.dim)
    print(np.min(frame_data),np.max(frame_data),np.unique(mask_data))
    
    input_train, input_val, input_test, label_train, label_val, label_test = \
                                train_test_val_split(frame_data,mask_data,name_list, opt.result_path)
                                
    n_train, n_test, n_val = len(input_train), len(input_test), len(input_val)
    print('***** #train: #test: #val = %d : %d :%d ******'%(n_train, n_test, n_val))

    Data_dict = {
            'train':[input_train.astype('float32'),
                     label_train.astype('float32')],
            'val':[input_val.astype('float32'),
                    label_val.astype('float32')],
            'test':[input_test.astype('float32'),
                    label_test.astype('float32')]
            }
    
    return Data_dict
