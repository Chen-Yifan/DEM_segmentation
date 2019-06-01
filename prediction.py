from custom_generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import model
from utils import *
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers.core import Dropout, Activation
from keras.models import Model,load_model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt

import keras.losses
import keras.metrics

model_name = 'Model_norm.h5'
result_name = 'Model_norm'

keras.losses.pixel_wise_loss = pixel_wise_loss
keras.metrics.Mean_IOU = Mean_IOU

test_frame_path = '/home/yifanc3/dataset/no_shuffle/test_frames'
test_mask_path = '/home/yifanc3/dataset/no_shuffle/test_masks'

Model_dir = '/home/yifanc3/models/norm_100epochs_59%'
# load model
m =load_model(os.path.join(Model_dir, model_name))

X,Y = test_gen(test_frame_path, test_mask_path)

score = m.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
# print("%s: %.2f%%" % (m.metrics_names[2], score[2]*100))
# print("%s: %.2f%%" % (m.metrics_names[3], score[3]*100))
# print("%s: %.2f%%" % (m.metrics_names[4], score[4]*100))
# print("%s: %.2f%%" % (m.metrics_names[5], score[5]*100))
# print("%s: %.2f%%" % (m.metrics_names[6], score[6]*100))

results = m.predict(X)
new_r = np.argmax(results,axis=-1)

#save image
# saveMask_256("/home/yifanc3/results/resplit_orig_mask",test_mask_path,Y)
result_path = os.path.join("/home/yifanc3/results/", result_name)

if not os.path.isdir(result_path):
    os.makedirs(result_path)
    
saveResult(result_path, test_mask_path,results)