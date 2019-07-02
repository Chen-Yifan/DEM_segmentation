from k_fold import *
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
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt

import keras.losses
import keras.metrics

Model_name = '128overlap_300w_segnetAdal_60ep_6c'
weights_name = 'weights.36-2.08-0.56.hdf5'
fold = 1
date = '6.27'

path = '/home/yifanc3/results/%s/%s/%s/'%(date,Model_name,fold)
test_mask_path = path + 'mask'
test_frame_path = path + 'frame'

shape = 128

def load_test(img_folder, mask_folder, shape=128):
    n = os.listdir(img_folder)
    img = np.zeros((len(n), shape, shape, 6)).astype(np.float32)
    mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)
    
    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        if(train_img_0.shape!=(shape,shape,6)):
            continue
        img[i] = train_img_0 #add to array - img[0], img[1], and so on.
        
        #train_mask
        train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
        mask[i] = train_mask
    return img, mask    

def saveResult(save_path, test_mask_path, results, flag_multi_class = False, num_class = 2, shape=128):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    print(save_path)
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

keras.losses.pixel_wise_loss = pixel_wise_loss
keras.metrics.Mean_IOU = Mean_IOU
keras.metrics.recall = recall
keras.metrics.precision = precision
keras.metrics.f1score = f1score
keras.metrics.per_pixel_acc = per_pixel_acc
from keras.utils import CustomObjectScope


Model_dir = '/home/yifanc3/models/%s/%s/ckpt_weights/%s/%s'%(date, Model_name, fold, weights_name)
# load model?

m = model.segnet(input_shape = (128,128,6))
m.load_weights(Model_dir)

opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)    
m.compile( optimizer = opt2, loss = pixel_wise_loss, metrics = [per_pixel_acc, Mean_IOU, precision, recall, f1score])

X,Y = load_test(test_frame_path, test_mask_path)

score = m.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
print("%s: %.2f%%" % (m.metrics_names[2], score[2]*100))
print("%s: %.2f%%" % (m.metrics_names[3], score[3]*100))
print("%s: %.2f%%" % (m.metrics_names[4], score[4]*100))
print("%s: %.2f%%" % (m.metrics_names[5], score[5]*100))
# print("%s: %.2f%%" % (m.metrics_names[6], score[6]*100))

results = m.predict(X)
new_r = np.argmax(results,axis=-1)

#save image
# saveMask_256("/home/yifanc3/results/v2_orig_mask",test_mask_path,Y)
result_path = path + weights_name[0:-5]+'-iou%.2f-results'%(score[2]*100)

if not os.path.isdir(result_path):
    os.makedirs(result_path)
saveResult(result_path,test_mask_path,results)