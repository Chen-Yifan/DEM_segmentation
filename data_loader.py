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
    
    return (np.sum(input_array)>0)


def load_feature_data(frame_dir, mask_dir, gradient=False, dim=512,shuffle=False):
    
    '''load frames and masks into two numpy array respectively
        -----
        condition: with feature
        input: frame_dir, mask_dir (each file in tif format)
        
        process: always resize to 128x128 as model input
        -----
        
    '''
    frames = []
    masks = []
    minn = float("inf")
    maxx = 0.0
    frame_names = os.listdir(frame_dir)
    if shuffle:
        random.shuffle(frame_names)
    else:
        frame_names.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    for i in range(len(frame_names)):
        print(i)
        if len(frames)>=270:
            break
        frame_file = frame_names[i]
        frame_path = os.path.join(frame_dir, frame_file)
    #tif
        mask_path = os.path.join(mask_dir, frame_file.replace('DEM','label'))
        frame_array = np.array(Image.open(frame_path))
        label_array = np.array(Image.open(mask_path))
    #npy
#         mask_path = os.path.join(mask_dir, frame_file)
#         frame_array = np.load(frame_path)
#         label_array = np.load(mask_path)
        dims = frame_array.shape
        if dims[0]!=dim or dims[1]!=dim:
            os.remove(mask_path)
            os.remove(frame_path)
            continue
        if(is_feature_present(label_array)):
            frame_array = np.array(Image.fromarray(frame_array).resize((128,128), Image.BILINEAR))
            label_array = np.array(Image.fromarray(label_array).resize((128,128), Image.NEAREST))
            if (len(np.unique(frame_array))<3):
                os.remove(mask_path)
                os.remove(frame_path) 
                continue
            if gradient:
                [dx, dy] = np.gradient(frame_array)
                frame_array = np.sqrt((dx*dx)+(dy*dy))
            frames.append(frame_array)
            masks.append(label_array)
        else:
            os.remove(mask_path)
            os.remove(frame_path)
            
    print(len(frames), len(masks))
    return np.array(frames),np.array(masks), minn, maxx

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
        
        Data[key][0] = Data[key][0].reshape(len(Data[key][0]), 128, 128, 1)
        Data[key][1] = Data[key][1].reshape(len(Data[key][1]), 128, 128, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.
            # img[img > 0.] = 1. - img[img > 0.]      #inv color
            minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
            img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = img 
            
