from k_fold import *
from losses import *
from metrics import *
from utils import *
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
from keras.optimizers import Adadelta, Adam, SGD
import matplotlib.pyplot as plt
from keras.models import model_from_json

import keras.losses
import keras.metrics

Model_name = '128over_MT3_unet_weightedloss_bce_9cdist_25e'
weights_name = 'weights.25-16.54-0.56.hdf5'
fold = 0
date = '8.14'
network = 'unet'
band = 6

path = '/home/yifanc3/results/%s/%s/%s/'%(date,Model_name,fold)
test_mask_path = path + 'mask'
test_frame_path = path + 'frame'

shape = 128

def load_test(img_folder, mask_folder, shape=128, band=6):
    img_list = os.listdir(img_folder)
    mask_list = os.listdir(mask_folder)
    imgs = np.zeros((len(img_list), shape, shape, band)).astype(np.float32)
    masks = np.zeros((len(img_list), shape, shape, 2), dtype=np.float32)
    masks_dist = np.zeros((len(img_list), shape, shape, 2), dtype=np.float32)
    features = np.zeros((len(img_list),2),dtype=np.uint8)
    
    for i in range(len(img_list)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+img_list[i]) #normalization:the range is about -100 to 360
        imgs[i] = train_img_0 #add to array - img[0], img[1], and so on.
        
        print(mask_list[2*i], mask_list[2*i+1])
        #train_mask
        binary = np.load(mask_folder+'/'+mask_list[2*i]) # 1.0 or 2.0 
        distance = np.load(mask_folder+'/'+mask_list[2*i+1])
        
        if len(np.unique(binary)) == 2:
            features[i:1] = 1 # has feature
        else:
            features[i:0] = 1 # no feature
            
        masks[i] = binary
        masks_dist[i] = distance
    return imgs, masks, masks_dist, features


def saveResult(save_path, test_frame_path, results,shape=128):
    mkdir(save_path)
    print(save_path)
    
    n = os.listdir(test_frame_path)
    length = len(n)
    
    distance_shape = results[0].shape
    binary_shape = results[1].shape
    
    clssify_pred = np.argmax(results[2],axis = -1)
    clssify_gt = np.argmax(test_y[2],axis = -1)
    np.save(os.path.join(save_path,'classification_gt.npy'), clssify_gt)
    np.save(os.path.join(save_path,'classification_pred.npy'), clssify_pred)
    
    for i in range(length):
        img_distance = np.argmax(results[0][i],axis = -1)
        img_binary = np.argmax(results[1][i],axis = -1)
        np.save(os.path.join(save_path,"%s_distance.npy"%name[0:-4]),img_distance)
        np.save(os.path.join(save_path,"%s_binary.npy"%name[0:-4]),img_binary)
    
    

keras.losses.pixel_wise_loss = pixel_wise_loss
keras.metrics.Mean_IOU = Mean_IOU
keras.metrics.Mean_IOU_dist = Mean_IOU_dist
keras.metrics.recall = recall
keras.metrics.precision = precision
keras.metrics.f1score = f1score
keras.metrics.accuracy = accuracy
keras.metrics.FP = FP
keras.metrics.FN = FN
keras.metrics.per_pixel_acc = per_pixel_acc
keras.losses.weighted_categorical_crossentropy = weighted_categorical_crossentropy

from keras.utils import CustomObjectScope


Model_dir = '/home/yifanc3/models/%s/%s/ckpt_weights/%s/%s'%(date, Model_name, fold, weights_name)
# load model?

#model 
# if(network == 'unet'):
#     m = model.get_unet_multitask(input_shape = (shape,shape,band))
# else:
#     m = model.segnet(multi_task = True, input_shape = (shape,shape,band))
json_path = '/home/yifanc3/models/%s/%s/model%s.json' %(date, Model_name, fold)
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
m = model_from_json(loaded_model_json, custom_objects = 
                    {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D})
m.load_weights(Model_dir)

opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
opt3 = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)

b_weights = np.array([1.0,300.0])
b_loss = weighted_categorical_crossentropy(b_weights)
d_weights = np.array([1.0,5.0,25.0,50.0,150.0])
d_loss = weighted_categorical_crossentropy(d_weights)
m.compile( 
          optimizer = opt2, 
          loss = {'binary':b_loss, 'distance': d_loss, 
                  'classification':'categorical_crossentropy'}, 
          loss_weights = {'binary':0.4, 'distance':0.3,'classification':0.3}, 
          metrics = {'binary':[per_pixel_acc, Mean_IOU, Mean_IOU_label, precision, recall, f1score], 
                      'distance': Mean_IOU_dist, 'classification':[accuracy, FN, FP]}
         )

test_x, test_y, test_y2, test_y3 = load_test(test_frame_path, test_mask_path, shape, band)

score = m.evaluate(test_x, {'binary': test_y, 'distance': test_y2, 'classification':test_y3}, verbose=0)

for i in range(14):
    print("%s: %.2f%%" % (m.metrics_names[i+1], score[i+1]*100))

results = m.predict(test_x)

#save image
# saveMask_256("/home/yifanc3/results/v2_orig_mask",test_mask_path,Y)
result_path = path + weights_name[0:-5]+'-iou%.2f-results'%(score[2]*100)

saveResult(result_path, test_frame_path, results, shape)
