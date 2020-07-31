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
    return (np.sum(input_array)>100)


def load_feature_data(frame_dir, mask_dir, gradient=False, dim=512,resize=False):
    
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
    minn = float("inf")
    maxx = 0.0
    frame_names = os.listdir(frame_dir)
    frame_names.sort(key=lambda var:[int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])
#     mask_names = os.listdir(mask_dir)
#     mask_names.sort(key=lambda var:[int(x) if x.isdigit() else x
#                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    print(frame_names[0], frame_names[1])
    
    for i in range(len(frame_names)):
        frame_file = frame_names[i]
        #if len(frames)>30:
        #    break
        frame_path = os.path.join(frame_dir, frame_file)
        mask_path = os.path.join(mask_dir, frame_file.replace('fillnodata','building_label'))
    #tif
        if(frame_file[-3:]=='tif'):
#             if not os.path.exists(mask_path):
#                 os.remove(frame_path)
#                 print('remove', frame_file)
#                 continue
            frame_array = np.array(Image.open(frame_path))
            label_array = np.array(Image.open(mask_path))    
    #npy
        else:
            mask_path = os.path.join(mask_dir, frame_file)
            frame_array = np.load(frame_path)[:,:,-1]
            label_array = np.load(mask_path)
        dims = frame_array.shape
        if dims[0]!=dim or dims[1]!=dim or (not is_feature_present(label_array)) or (len(np.unique(frame_array))<3):
#             os.remove(mask_path)
#             os.remove(frame_path)
            continue
        
        if gradient:
            [dx, dy] = np.gradient(frame_array)
            frame_array = np.sqrt((dx*dx)+(dy*dy))
        if resize:
            frame_array = np.array(Image.fromarray(frame_array).resize((128,128), Image.BILINEAR))
            label_array = np.array(Image.fromarray(label_array).resize((128,128), Image.NEAREST))
            
        name_list.append(frame_names[i])
        frames.append(frame_array)
        masks.append(label_array)
            
    print(len(frames), len(masks))
    return np.array(frames),np.array(masks),name_list, minn, maxx

def preprocess(Data, minn, maxx, dim=128, low=0.1, hi=1.0):
    """Normalize and rescale (and optionally invert) images.
    Parameters
    ----------
    Data : hdf5
        Data array.
    dim : integer, optional
        Dimensions of images, assumes square.
    low : float, optional
        Minimum rescale value. Default is 0.1 since background pixels are 0.
    hi : float, optional
        Maximum rescale value.
    """
    for key in Data:
        print (key)
        if Data[key][0].ndim != 4:
            Data[key][0] = Data[key][0].reshape(len(Data[key][0]), 128, 128, 1)
            
        Data[key][1] = Data[key][1].reshape(len(Data[key][1]), 128, 128, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.
            # img[img > 0.] = 1. - img[img > 0.]      #inv color
            minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
            img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = img 
            
