from k_fold import *
from utils import *
from metrics import *
from losses import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import model
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.layers.core import Dropout, Activation
from keras.models import Model,load_model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt

import keras.losses
import keras.metrics

Model_name = '128overlap_MT_segnet_45epoch'
weights_name = 'weights.36-2.35-0.57.hdf5'
fold = 1
date = '7.18'
# network = 'segnet'
band = 6
shape = 128

path = '/home/yifanc3/results/%s/%s/%s/'%(date,Model_name,fold)
test_mask_path = path + 'mask'
test_frame_path = path + 'frame'



def load_test(img_folder, mask_folder, shape=128, band=6):
    img_list = os.listdir(img_folder)
    mask_list = os.listdir(mask_folder)
    imgs = np.empty((len(img_list), shape, shape, band), dtype=np.float32)
    masks = np.empty((len(img_list), shape, shape, 2), dtype=np.float32)
    masks_dist = np.empty((len(img_list), shape, shape, 5), dtype=np.float32)
    
    print(len(img_list)) # 754
    for i in range(len(img_list)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+img_list[i]) #normalization:the range is about -100 to 360
        imgs[i] = train_img_0 #add to array - img[0], img[1], and so on.
        
        #train_mask
        binary = np.load(mask_folder+'/'+mask_list[2*i]) # 1.0 or 2.0 
        distance = np.load(mask_folder+'/'+mask_list[2*i+1])
        masks[i] = binary
        masks_dist[i] = distance
    return imgs, masks, masks_dist


def saveResult(save_path, test_frame_path, results,shape=128):
    mkdir(save_path)
    print(save_path)
    
    n = os.listdir(test_frame_path)
    length = len(n)
    
    distance_shape = results[0].shape
    binary_shape = results[1].shape
    
    for i in range(length):
        name = n[i]
        img_distance = np.argmax(results[0][i],axis = -1)
        img_binary = np.argmax(results[1][i],axis = -1)
        np.save(os.path.join(save_path,"%s_distance.npy"%name[0:-4]),img_distance)
        np.save(os.path.join(save_path,"%s_binary.npy"%name[0:-4]),img_binary)
    
    

keras.losses.pixel_wise_loss = pixel_wise_loss
keras.metrics.Mean_IOU = Mean_IOU
keras.metrics.Mean_IOU = Mean_IOU_dist
keras.metrics.recall = recall
keras.metrics.precision = precision
keras.metrics.f1score = f1score
keras.metrics.per_pixel_acc = per_pixel_acc
from keras.utils import CustomObjectScope
from keras.models import model_from_json
from layers import *


ckpt_dir = '/home/yifanc3/models/%s/%s/ckpt_weights/%s/%s'%(date, Model_name, fold, weights_name)
# load model?

#MODEL
# if(network == 'unet'):
#     m = model.get_unet_multitask(input_shape = (shape,shape,band))
# else:
#     m = model.segnet(multi_task = True, input_shape = (shape,shape,band))

# load json and create model
json_path = '/home/yifanc3/models/%s/%s/model%s.json' %(date, Model_name, fold)
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
m = model_from_json(loaded_model_json, custom_objects = 
                    {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D})

# ckpt_dir = '/home/yifanc3/models/%s/%s/model%s.h5' %(date, Model_name, fold)
m.load_weights(ckpt_dir)

print('====== LOAD WEIGTHS SUCCESSFULLY ======')

opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)    
#change to multimask loss!!
m.compile( optimizer = opt2, loss = {'binary':pixel_wise_loss, 'distance':'categorical_crossentropy'}, 
              loss_weights = {'binary':0.5, 'distance':0.5}, metrics = {'binary':[per_pixel_acc, Mean_IOU, 
                                                                                  precision, recall, f1score], 
                                                                        'distance':Mean_IOU_dist})

test_x, test_y, test_y2 = load_test(test_frame_path, test_mask_path, shape, band)

score = m.evaluate(test_x, {'binary': test_y, 'distance': test_y2}, verbose=0)

for i in range(8):
    print("%s: %.2f%%" % (m.metrics_names[i+1], score[i+1]*100))

results = m.predict(test_x)

#save image
# saveMask_256("/home/yifanc3/results/v2_orig_mask",test_mask_path,Y)
result_path = path + weights_name[0:-5]+'-iou%.2f-results'%(score[2]*100)

saveResult(result_path, test_frame_path, results, shape)
