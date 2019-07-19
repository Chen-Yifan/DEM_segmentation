from generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os
import tensorflow as tf
import keras.backend as K
import model
from metrics import *
from losses import *
from k_fold import *
import numpy as np
from keras.models import Model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
import time
from functools import *
from keras.models import model_from_json


#hyperparameters
date = 'tryout'
BATCH_SIZE = 32
NO_OF_EPOCHS = 45
shape = 128
aug = False # to decide if shuffle
Model_name = '128overlap_dist_1epoch'
network = 'segnet'
k = 2
band = 6
norm = True

print('batch_size:', BATCH_SIZE, '\ndate:', date, '\nshape:', shape, '\naug:',aug, '\nModel_name', Model_name, '\nk:',k, '; band:', band, '\nnorm:', norm)
    
#Train the model with K-fold Cross Val
#TRAIN
train_frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b/'
train_mask_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_10m6b/'
train_maskdst_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_10mdist/'


Model_path = '/home/yifanc3/models/%s/%s/' % (date,Model_name)
mkdir(Model_path)
    
Checkpoint_path = Model_path + 'ckpt_weights/'
mkdir(Checkpoint_path)


# k-fold cross-validation
img, mask, dstmask = load_data_multi(train_frame_path, train_mask_path, train_maskdst_path, shape, band, norm)
train_list, test_list = k_fold(len(img), k = k)
print(len(train_list), len(test_list))

model_history = [] 

for i in range(k):
    print('====The %s Fold===='%i)
    #shuffle the index
#     random.shuffle(train_list[i])
#     random.shuffle(test_list[i])
    
    train_x = img[train_list[i]]
    train_y = mask[train_list[i]]
    test_x = img[test_list[i]]
    test_y = mask[test_list[i]]
    
    train_y2 = dstmask[train_list[i]]
    test_y2 = dstmask[test_list[i]]
    
    #MODEL BUILD  multi_task
    if(network == 'unet'):
        m = model.get_unet_multitask(input_shape = (shape,shape,band))
    else:
        m = model.segnet(multi_task = True, input_shape = (shape,shape,band))
    
    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    m.compile( optimizer = opt2, loss = {'binary':pixel_wise_loss, 'distance':'categorical_crossentropy'}, 
              loss_weights = {'binary':0.5, 'distance':0.5}, metrics = {'binary':[per_pixel_acc, Mean_IOU, 
                                                                                  precision, recall, f1score], 
                                                                        'distance':Mean_IOU_dist})

    #callback
    ckpt_path = Checkpoint_path + '%s/'%i
    mkdir(ckpt_path)
    
    weights_path = ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_binary_Mean_IOU:.2f}.hdf5'
    
    callbacks = get_callbacks(weights_path, Model_path, 5)
    
    if(aug):
    # data augmentation
        train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_aug(train_x, {'binary': train_y, 
                                                                                              'distance': train_y2}, 
                                                                                    32, ratio = 0.18)
        history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data=val_gen,
                              validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                              shuffle = True,
                              callbacks=callbacks)
    else:
#         train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_noaug(train_x, train_y, 32, ratio = 0.18)
        history = m.fit(train_x, {'binary': train_y, 'distance': train_y2}, epochs=NO_OF_EPOCHS, 
                        batch_size=BATCH_SIZE, callbacks=callbacks,
                         verbose=1, validation_split=0.18, shuffle = True)
    
    model_history.append(history)
    
    # serialize model to JSON
    model_json = m.to_json()
    with open(os.path.join(Model_path,"model%s.json" %i), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("Saved model to disk")
    m.save(os.path.join(Model_path,'model%s.h5' %i))
    
  #TEST
    print('======Start Testing======')

    score = m.evaluate(test_x, {'binary': test_y, 'distance': test_y2}, verbose=0)
    for i in range(8):
        print("%s: %.2f%%" % (m.metrics_names[i+1], score[i+1]*100))
    

 #prediction
    results = m.predict(test_x)
    print(results[0].shape) # (n,128,128,5)

    #save image
    result_path = "/home/yifanc3/results/%s/%s/%s"%(date,Model_name,i)
    mkdir(result_path)

    print('result:', result_path)
    
    test_y = [test_y, test_y2]
    save_result(train_frame_path, result_path, test_list[i], results, test_x, test_y, shape, multi_task=True)
    # saveFrame_256(save_frame_path, test_frame_path, X)
    print("======="*12, end="\n\n\n")
