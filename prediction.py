import os
import numpy as np
from keras.callbacks import ModelCheckpoint,CSVLogger, EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD

import tensorflow as tf
from models import models
from metrics import *
from losses import *
from k_fold import *
from util.util import *

import keras.losses
import keras.metrics

Model_name = '128over5m6b_resnet_Adadelta_200w_65e'
weights_name = 'weights.33-11.76-0.55-0.10.hdf5'
fold = 1
date = '9.17'
network = 'resnet'
band = 5

path = '/home/yifanc3/results/%s/%s/%s/'%(date,Model_name,fold)
test_mask_path = path + 'mask'
test_frame_path = path + 'frame'

shape = 128

def load_test(img_folder, mask_folder, shape=128, band=5):
    n = os.listdir(img_folder)
    img = np.zeros((len(n), shape, shape, band)).astype(np.float32)
    mask = np.zeros((len(n), shape, shape, 2), dtype=np.float32)
    
    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        if(train_img_0.shape!=(shape,shape,band)):
            continue
        img[i] = train_img_0 #add to array - img[0], img[1], and so on.
        
        #train_mask
        train_mask = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0 
        mask[i] = train_mask
    return img, mask    

def saveResult(save_path, test_mask_path, results,shape=128, num_class = 2):
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

keras.losses.weighted_categorical_crossentropy = weighted_categorical_crossentropy
keras.metrics.Mean_IoU_cl = Mean_IoU_cl
keras.metrics.Mean_IOU_label = Mean_IOU_label
keras.metrics.recall = recall
keras.metrics.precision = precision
keras.metrics.f1score = f1score
keras.metrics.per_pixel_acc = per_pixel_acc
from keras.utils import CustomObjectScope

Model_path = '/home/yifanc3/models/%s/%s/'%(date, Model_name)
Model_dir = '/home/yifanc3/models/%s/%s/ckpt_weights/%s/%s'%(date, Model_name, fold, weights_name)
# load model?

#model 
input_shape = (shape,shape,band)
if(network == 'unet'):
    m = models.unet(input_shape = input_shape)
elif(network == 'segnet'):
    m = models.segnet(input_shape = input_shape)
else:
    m = models.resnet(input_shape=input_shape)
    
m.load_weights(Model_dir)

opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)    

weights = np.array([1.0,200.0])
loss = weighted_categorical_crossentropy(weights) 
Mean_IOU = Mean_IoU_cl(cl=2)
m.compile( optimizer = opt2, loss = loss, metrics = [per_pixel_acc, Mean_IOU, Mean_IOU_label, precision, recall, f1score])

X,Y = load_test(test_frame_path, test_mask_path, shape, band)
score = m.evaluate(X, Y, verbose=0)

message = ''
for j in range(7):
    print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
    message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)
    
output_file = os.path.join(Model_path, 'output_%s_%s'%(fold, weights_name[8:10]))
with open(output_file, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
        

results = m.predict(X)
new_r = np.argmax(results,axis=-1)

#save image
# saveMask_256("/home/yifanc3/results/v2_orig_mask",test_mask_path,Y)
result_path = path + weights_name[0:-5]+'-iou%.2f-results'%(score[2]*100)

mkdir(result_path)
saveResult(result_path,test_mask_path,results,shape)
