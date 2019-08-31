import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
from keras.utils import to_categorical
from itertools import product


def saveResult(save_path, test_mask_path, results, flag_multi_class = False, num_class = 2, shape=128):
    n = os.listdir(test_mask_path)
    result_shape = np.shape(results)
    print(result_shape)
    results = results.reshape(len(n),shape,shape,2)
    #results = results.astype('uint8')
    for i in range(result_shape[0]):
        img = np.argmax(results[i],axis = -1)
        img = np.squeeze(img)
        #cv2.imwrite(os.path.join(save_path,"%s_predict.png"%n[i][0:-4]),results[i])
        np.save(os.path.join(save_path,"%s_predict.npy"%n[i][0:-4]),img)
        
        
def saveMask_256(save_path, test_mask_path, test_mask):
    n = os.listdir(test_mask_path)
    shape = np.shape(test_mask)
    test_mask = test_mask.reshape(len(n),shape[1],shape[2],2)
    for i in range(len(n)):
        img = np.argmax(test_mask[i],axis = -1)
        np.save(os.path.join(save_path,"%s"%n[i]),img)
        
def saveFrame_256(save_path, test_frame_path, test_frame):
    n = os.listdir(test_frame_path)
    for i in range(len(n)):
        np.save(os.path.join(save_path,"%s"%n[i]),test_frame[i])



def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
