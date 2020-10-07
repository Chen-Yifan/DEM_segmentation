from PIL import Image
import numpy as np
import os
import re
import scipy.misc
import random

def is_feature_present(input_array):
    # num_1 = np.count_nonzero(input_array)
    # num_0 = 512*512 - num_1
    # print('********point1***********\n label #1 > #0', num_1>num_0)
    # assert num_1 < num_0, 'label 1 should be label but now is background'
#     print(np.sum(input_array))
    return (np.sum(input_array)>150)


def load_feature_data(frame_dir, mask_dir, dim=128, gradient=False, resize=False):
    
    '''load frames and masks into two numpy array respectively
        -----
        condition: with feature
        input: frame_dir, mask_dir (each file in tif format)
        
        process: always resize to 128x128 as model input
        -----
        
    '''
    frames = []
    masks = []
    name_list = []
    frame_names = os.listdir(frame_dir)
    frame_names.sort(key=lambda var:[int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])
#     mask_names = os.listdir(mask_dir)
#     mask_names.sort(key=lambda var:[int(x) if x.isdigit() else x
#                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    print(frame_names[0], frame_names[1])
    
    for i in range(len(frame_names)):
        frame_file = frame_names[i]
        # if len(frames)>1000:
            # break
        frame_path = os.path.join(frame_dir, frame_file)
        # mask_path = os.path.join(mask_dir, frame_file.replace('fillnodata','building_label'))
        #### for 128_0ver
        mask_path = os.path.join(mask_dir, frame_file.replace('DEM','label'))
    #tif
        if(frame_file[-3:]=='tif'):
            if not os.path.exists(mask_path):
                # os.remove(frame_path)
                # print('remove', frame_file)
                continue
            frame_array = np.array(Image.open(frame_path))
            label_array = np.array(Image.open(mask_path))    
    #npy
        else:
            mask_path = os.path.join(mask_dir, frame_file)
            x = np.load(frame_path)
            # frame_array = np.concatenate((x[:,:,0:2], np.expand_dims(x[:,:,-1], axis=2)),axis=-1)
            frame_array = x[:,:,-1]
            label_array = np.load(mask_path)
        dims = frame_array.shape
        if dims[0]!=dim or dims[1]!=dim or (len(np.unique(frame_array))<3): # or  (not is_feature_present(label_array)) 
            # os.remove(mask_path)
            # os.remove(frame_path)
            continue

        if  (not is_feature_present(label_array)):
            continue
        # if gradient:
        #     [dx, dy] = np.gradient(frame_array)
        #     frame_array = np.sqrt((dx*dx)+(dy*dy))

        if resize:
            frame_array = np.array(Image.fromarray(frame_array).resize((128,128), Image.BILINEAR))
            label_array = np.array(Image.fromarray(label_array).resize((128,128), Image.NEAREST))
            
        name_list.append(frame_names[i])
        frames.append(frame_array)
        masks.append(label_array)
            
    print(len(frames), len(masks))
    frames, masks = np.array(frames), np.array(masks)
    # reshape to 4 dimensions
    if(frames.ndim != 4):
        frames = frames.reshape((len(frames), dim, dim, 1))
    masks = masks.reshape((len(masks),dim, dim, 1))
    return frames, masks, name_list

def preprocess(Data, dim=128, low=0.1, hi=1.0):
    """Normalize and rescale (and optionally invert) images.
    Parameters
    ----------
    Data : hdf5
        Data array.
    minn : list of minimum of each band
    maxx : list of maximum of each band
    dim : integer, optional
        Dimensions of images, assumes square.
    low : float, optional
        Minimum rescale value. Default is 0.1 since background pixels are 0.
    hi : float, optional
        Maximum rescale value.
    """
    bands = Data['train'][0].shape[-1]
    print('#bands, ', bands)
    for key in Data:
        print (key)
        for i, imgs in enumerate(Data[key][0]):
            # imgs = imgs / 255.
            # img[img > 0.] = 1. - img[img > 0.]      #inv color
            for b in range(bands):
                img = imgs[:,:,b]
                minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
                img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = imgs 
            
