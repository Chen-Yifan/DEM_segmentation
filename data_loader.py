from libtiff import TIFF
import numpy as np
import os

def is_feature_present(input_array):
    num_1 = np.count_nonzero(input_array)
    num_0 = 512*512 - num_1
    print('********point1***********\n label #1 > #0', num_1>num_0)
    assert num_1 < num_0, 'label 1 should be label but now is background'
    
    return (np.sum(input_array)>0)


def load_feature_data(frame_dir, mask_dir, gradient=True):
    
    '''load frames and masks into two numpy array respectively
        -----
        condition: with feature
        input: frame_dir, mask_dir (each file in tif format)
        -----
    '''
    frames = []
    masks = []
    minn = float("inf")
    maxx = 0.0
    
    frame_names = os.listdir(frame_dir)
    for frame_file in frame_names:
        frame_path = os.path.join(frame_dir, frame_file)
        mask_path = os.path.join(mask_dir, frame_file.replace('fillnodata','building_label'))
        input_array = TIFF.open(frame_path).read_image()
        label_array = TIFF.open(mask_path).read_image()
        
        if(is_feature_present(label_array)):
            if gradient:
                frame_array = np.gradient(frame_array)
                label_array = np.gradient(label_array)
                amin, amax = np.min(frame_array), np.max(frame_array)
                if amin < minn: minn = amin 
                if amax > maxx: maxx = amax 
            frames.append(frame_array)
            masks.append(label_array)
           
    return np.array(frames),np.array(masks), minn, maxx


def preprocess(Data, minn, maxx, width=512, low=0.1, hi=1.):
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

        Data[key][0] = Data[key][0].reshape(len(Data[key][0]), dim, dim, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.
            # minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
            img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = img 
    
