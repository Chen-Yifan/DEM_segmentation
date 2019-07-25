from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import keras.backend as K
import model
from utils import *
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import time
from functools import *
import random
import re
from keras.callbacks import TensorBoard


def get_callbacks(name_weights, path, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=False, monitor='Mean_IOU', mode='max')
#     reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    logdir = os.path.join(path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    return [mcp_save, tensorboard]

def save_result(train_frame_path, save_path, test_idx, results, test_x, test_y, shape = 128, multi_task = False):
    n = os.listdir(train_frame_path)
    a = test_idx[0]
    b = test_idx[-1]
    length = len(test_idx)
    print(a,b)
    print(length)
    save_frame_path = os.path.join(save_path, 'frame')
    save_mask_path = os.path.join(save_path, 'mask')
    save_result_path = os.path.join(save_path, 'prediction')
    if not os.path.isdir(save_result_path):
        os.makedirs(save_result_path)
    if not os.path.isdir(save_frame_path):
        os.makedirs(save_frame_path)
    if not os.path.isdir(save_mask_path):
        os.makedirs(save_mask_path)
        
    if(multi_task):
        distance_shape = results[0].shape
        binary_shape = results[1].shape
        #save features
        np.save(os.path.join(save_path, 'features_pred.npy'), np.argmax(results[2], axis=-1))
        for i in range(length):
            name = n[a+i]
            img_distance = np.argmax(results[0][i],axis = -1)
            img_binary = np.argmax(results[1][i],axis = -1)
            np.save(os.path.join(save_result_path,"%s_distance.npy"%name[0:-4]),img_distance)
            np.save(os.path.join(save_result_path,"%s_binary.npy"%name[0:-4]),img_binary)

            np.save(os.path.join(save_frame_path,"%s.npy"%name[0:-4]),test_x[i])
            np.save(os.path.join(save_mask_path,"%s_binary.npy"%name[0:-4]),test_y[0][i])
            np.save(os.path.join(save_mask_path,"%s_distance.npy"%name[0:-4]),test_y[1][i])
    else:
        result_shape = np.shape(results)
        print(result_shape)
#         results = results.reshape(length,shape,shape,2)
        #results = results.astype('uint8')
        for i in range(result_shape[0]):
            name = n[a+i]
            img = np.argmax(results[i],axis = -1)
            img = np.squeeze(img)
            np.save(os.path.join(save_result_path,"%s_predict.npy"%name[0:-4]),img)

            np.save(os.path.join(save_frame_path,"%s.npy"%name[0:-4]),test_x[i])
            np.save(os.path.join(save_mask_path,"%s.npy"%name[0:-4]),test_y[i])

        
def load_data_multi(img_folder, mask_folder, maskdist_folder, shape=128, band=6, norm=True):
    n = os.listdir(img_folder)
    n.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
#     random.shuffle(n)
    
    if(band == 6):
        img = np.zeros((len(n), shape, shape, 6)).astype(np.float32)
        mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)
        mask_dist = np.zeros((len(n), shape, shape, 5), dtype=np.float32)
        features = np.zeros((len(n),2),dtype=np.uint8)

        for i in range(len(n)): #initially from 0 to 16, c = 0. 
            train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
            if(train_img_0.shape!=(shape,shape,6)):
                continue
#             train_img_0 = np.where(train_img_0==-9999, 0.0, train_img_0)
            #interpolate all negative values
            
            if(norm):
                train_img_0[:,:,0] = train_img_0[:,:,0] / 88

                #mclean_roi_aspect
                train_img_0[:,:,1] = (train_img_0[:,:,1]) / 360

                #mclean_roi_rough
                train_img_0[:,:,2] = train_img_0[:,:,2] / 313
#                 train_img_0[:,:,2] = np.where(train_img_0[:,:,2]<0, -1, train_img_0[:,:,2])
                #mclean_roi_tpi
                train_img_0[:,:,3] = (train_img_0[:,:,3]) / (275)

                #mclean_roi_tri
                train_img_0[:,:,4] = train_img_0[:,:,4] / 305
#                 train_img_0[:,:,4] = np.where(train_img_0[:,:,4]<0, -1, train_img_0[:,:,4])
                #mclean_roi
                train_img_0[:,:,-1] = train_img_0[:,:,-1] / 1174
#                 train_img_0[:,:,-1] = np.where(train_img_0[:,:,-1]<0, -1, train_img_0[:,:,-1])
            
            else:
                train_img_0[:,:,-1] = np.where(train_img_0[:,:,-1]<0, -1, train_img_0[:,:,-1])
                train_img_0[:,:,-1] = train_img_0[:,:,-1] - 640
                
            img[i] = train_img_0
         

            #train_mask
            train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
            if len(np.unique(train_mask)) == 2:
                features[i,1] = 1 # has feature
            else:
                features[i,0] = 1 # no feature
                
            mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
            mask[i,:,:,1] = np.squeeze(train_mask)
            
            train_mask_dist = np.load(maskdist_folder+'/'+n[i])
            train_mask_dist = np.eye(5)[train_mask_dist]
            mask_dist[i] = train_mask_dist
            
        return img, mask, mask_dist, features
        
def load_data(img_folder, mask_folder, shape=128, band=5, norm=True):
    n = os.listdir(img_folder)
    n.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
#     random.shuffle(n)
    
    if(band == 6):
        img = np.zeros((len(n), shape, shape, 6)).astype(np.float32)
        mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)
        mask_dist = np.zeros((len(n), shape, shape, 5), dtype=np.float32)

        for i in range(len(n)): #initially from 0 to 16, c = 0. 
            train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
            if(train_img_0.shape!=(shape,shape,6)):
                continue
#             train_img_0 = np.where(train_img_0==-9999, 0.0, train_img_0)
            #mclean_roi_slope
            if(norm):
                train_img_0[:,:,0] = train_img_0[:,:,0] / 88

                #mclean_roi_aspect
                train_img_0[:,:,1] = (train_img_0[:,:,1]) / 360

                #mclean_roi_rough
                train_img_0[:,:,2] = train_img_0[:,:,2] / 313
                train_img_0[:,:,2] = np.where(train_img_0[:,:,2]<0, -1, train_img_0[:,:,2])
                #mclean_roi_tpi
                train_img_0[:,:,3] = (train_img_0[:,:,3]+9999) / (9999+275)

                #mclean_roi_tri
                train_img_0[:,:,4] = train_img_0[:,:,4] / 305
                train_img_0[:,:,4] = np.where(train_img_0[:,:,4]<0, -1, train_img_0[:,:,4])
                #mclean_roi
                train_img_0[:,:,-1] = train_img_0[:,:,-1] / 1174
                train_img_0[:,:,-1] = np.where(train_img_0[:,:,-1]<0, -1, train_img_0[:,:,-1])
            
            else:
                train_img_0[:,:,-1] = np.where(train_img_0[:,:,-1]<0, -1, train_img_0[:,:,-1])
                train_img_0[:,:,-1] = train_img_0[:,:,-1] - 640
                
            img[i] = train_img_0
         

            #train_mask
            train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
            mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
            mask[i,:,:,1] = np.squeeze(train_mask)
            
    elif band==5:
        print('band is 5')
        img = np.zeros((len(n), shape, shape, 5)).astype(np.float32)
        mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)

        for i in range(len(n)): #initially from 0 to 16, c = 0. 
            train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
            if(train_img_0.shape!=(shape,shape,6)):
                continue
            train_img = train_img_0[:,:,0:5]
            img[i] = train_img  #add to array - img[0], img[1], and so on.
#             img[i,:,:,5] -= 640
            #train_mask
            train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
            mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
            mask[i,:,:,1] = np.squeeze(train_mask)
    elif band == 1:
        img = np.zeros((len(n), shape, shape, 1)).astype(np.float32)
        mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)

        for i in range(len(n)): #initially from 0 to 16, c = 0. 
            train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
            if(train_img_0.shape!=(shape,shape,6)):
                continue
            
            img[i] = np.expand_dims((train_img_0[:,:,1]) / 360, axis = -1) #add to array - img[0], img[1], and so on.

            #train_mask
            train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
            mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
            mask[i,:,:,1] = np.squeeze(train_mask)            
    
    return img, mask

def k_fold(n, k=3):
  #shuffle the training one ?
    # ratio = 1/10s
    idx = list(np.arange(n))
    train_list = []
    test_list = []
    print('k_fold')
    for i in range(k):
#         i = k - i - 1
        a = int(i*n*3/20)
        b = int((i+1)*n*3/20)
        print('fold', i,' test ', 'start:', a,'end:',b)
        test_index = idx[a:b]
        train_index = idx[:a] + idx[b:]
        test_list.append(np.array(test_index))
        train_list.append(np.array(train_index))
    return train_list, test_list

# data augmentation
def train_gen_aug(img_list, mask_list, batch_size=32, ratio = 0.18):
    
    n = len(img_list)
    a = int(n*(1-0.18))
    b = n - a
    
    data_gen_args = dict(
                    horizontal_flip = True,
#                      vertical_flip = True,
                     width_shift_range = 0.1,
                     height_shift_range = 0.1,
                     zoom_range = 0.2, #resize
                     rotation_range = 10,
#                      featurewise_center=True,
    )
    
    
    train_img = img_list[0:a]
    train_mask = mask_list[0:a]
    val_img = img_list[a:]
    val_mask = mask_list[a:]
# train gen
    img_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2018
    img_gen = img_datagen.flow(train_img, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = img_datagen.flow(train_mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

# val_gen
    img_datagen = ImageDataGenerator()

    img_gen = img_datagen.flow(val_img, batch_size=batch_size, shuffle=True)
    mask_gen = img_datagen.flow(val_mask, batch_size=batch_size, shuffle=True)
    val_gen = zip(img_gen, mask_gen)    
    
    return train_gen, val_gen, a, b

def MTgenerator(img_list, mask_list, mask_dist_list, features_list, split, batch_size=32):
    if(split == 'train'):
        data_gen_args = dict(
                    horizontal_flip = True,
                     width_shift_range = 0.1,
                     height_shift_range = 0.1,
                     zoom_range = 0.2, #resize
                     rotation_range = 10,
                )
        img_datagen = ImageDataGenerator(**data_gen_args)
        
    else:
        img_datagen = ImageDataGenerator()

    seed = 2018
    img_gen = img_datagen.flow(img_list, features_list, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = img_datagen.flow(mask_list, seed = seed, batch_size=batch_size, shuffle=True)
    mask_dist_gen = img_datagen.flow(mask_dist_list, seed = seed, batch_size=batch_size, shuffle=True)
       
    
    #yield generated data
    while True:
        X, Y3 = img_gen.next()
        Y1 = mask_gen.next()
        Y2 = mask_dist_gen.next()
        yield X, [Y2, Y1, Y3]

        
