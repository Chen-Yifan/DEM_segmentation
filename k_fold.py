from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
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


def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]

def save_result(train_frame_path, save_path, test_idx, results, test_x, test_y):
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
        
    shape = np.shape(results)
    print(shape)
    results = results.reshape(length,128,128,2)
    #results = results.astype('uint8')
    for i in range(shape[0]):
        name = n[a+i]
        img = np.argmax(results[i],axis = -1)
        img = np.squeeze(img)
        np.save(os.path.join(save_result_path,"%s_predict.npy"%name[0:-4]),img)
        
        np.save(os.path.join(save_frame_path,"%s.npy"%name[0:-4]),test_x[i])
        np.save(os.path.join(save_mask_path,"%s.npy"%name[0:-4]),test_y[i])


def load_data(img_folder, mask_folder):
    n = os.listdir(img_folder)
    random.shuffle(n)
    
#     img = np.zeros((len(n), 256, 256, 5)).astype(np.float32)
#     mask = np.zeros((len(n), 256, 256, 2), dtype=np.float32)
    
#     for i in range(len(n)): #initially from 0 to 16, c = 0. 
#         train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
#         #train_img =  cv2.resize(train_img, (256, 256))# Read an image from folder and resize
#         train_img = np.zeros((256,256,5))
#         #resize
#         for a in range(5):
#             train_img[:,:,a] = cv2.resize(train_img_0[:,:,a], (256, 256))
#         img[i] = train_img #add to array - img[0], img[1], and so on.
        
#         #train_mask
#         train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
# #         train_mask = np.where(train_mask_0==2.0, 0.0, train_mask_0) 
#         #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
#         train_mask = cv2.resize(train_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
#         mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
#         mask[i,:,:,1] = np.squeeze(train_mask)


#64 64
    img = np.zeros((len(n), 128, 128, 5)).astype(np.float32)
    mask = np.zeros((len(n), 128, 128, 2), dtype=np.float32)
    
    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        if(train_img_0.shape!=(128,128,5)):
            continue
        img[i] = train_img_0 #add to array - img[0], img[1], and so on.
        
        #train_mask
        train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
        mask[i,:,:,0] = np.squeeze(1-train_mask) # 0 to 1
        mask[i,:,:,1] = np.squeeze(train_mask)
    return img, mask

def k_fold(n, k=3):
    # ratio = 2/10s
    idx = np.arange(n)
    train_list = []
    val_list = []
    print('k_fold')
    for i in range(k):
        a = int(i*n/5)
        b = int((i+1)*n/5)
        print('fold', i,' val ', 'start:', a,'end:',b)
        
        val_list.append(idx[a:b])
        train_idx_array = np.append(idx[:a], idx[b:])
        train_list.append(train_idx_array)
    print(len(val_list))
    return train_list, val_list

def cv_data_gen(IMG, MASK, train_list, batch_size, j):
    c = 0
    array = train_list[j] #List of training images
    n = len(array)
    random.shuffle(array)
    train_img = IMG[train_list[j]]
    train_mask = MASK[train_list[j]]
#     val_img = IMG[val_list[i]]
#     val_mask = MASK[val_list[i]]
    
    while (True):
        img = np.zeros((batch_size, 128, 128, 5)).astype(np.float32)
        mask = np.zeros((batch_size, 128, 128, 2), dtype=np.float32)
        
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            img[i-c] = train_img[i]
            mask[i-c] = train_mask[i]
            
        mask = mask.reshape(batch_size,IMAGE_H*IMAGE_W, 2)
        
        c+=batch_size
        if(c+batch_size>=n):
            c=0
            random.shuffle(array)
        yield img, mask 

# data augmentation
def train_gen_aug(img_folder, mask_folder, batch_size):
    
#     datagen = ImageDataGenerator(
#     featurewise_center=True,
# #     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
    data_gen_args = dict(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10
                        )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    
    img = np.zeros((len(n), 256, 256, 5)).astype(np.float32)
    mask = np.zeros((len(n), 256, 256, 2), dtype=np.float32)

    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        #train_img =  cv2.resize(train_img, (256, 256))# Read an image from folder and resize
        train_img = np.zeros((256,256,5))
        #resize
        for a in range(5):
            train_img[:,:,a] = cv2.resize(train_img_0[:,:,a], (256, 256))
        img[i] = train_img #add to array - img[0], img[1], and so on.
        
        #train_mask
        train_mask_0 = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0
        train_mask = np.where(train_mask_0==2.0, 0.0, train_mask_0) 
        #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
        train_mask = cv2.resize(train_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
        mask[i,:,:,0] = np.squeeze(train_mask)
        mask[i,:,:,1] = np.squeeze(1-train_mask)
        
    seed = 2018

    
    img_gen = image_datagen.flow(img, seed = seed, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

        
    return train_gen