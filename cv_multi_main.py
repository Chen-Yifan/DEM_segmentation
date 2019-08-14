# MT3
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
from keras.optimizers import Adadelta, Adam, SGD
import matplotlib.pyplot as plt
import time
from functools import *
from keras.models import model_from_json


#hyperparameters
date = '8.13'
BATCH_SIZE = 32
NO_OF_EPOCHS = 80
shape = 128
aug = False
Model_name = '128over_MT3_segnet_weightedloss_10mbinary_80e'
network = 'segnet'
k = 2
band = 6
norm = True

print('batch_size:', BATCH_SIZE, '\ndate:', date, '\nshape:', shape, '\naug:',aug, '\nNetwork:', network,'\nModel_name:', Model_name, '\nk:',k, '; band:', band, '\nnorm:', norm)
    
#Train the model with K-fold Cross Val
#TRAIN
train_frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b_norm/'
train_mask_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_10m6b/'
train_maskdst_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_10mdist/'

print(train_frame_path, train_mask_path, train_maskdst_path)

Model_path = '/home/yifanc3/models/%s/%s/' % (date,Model_name)
mkdir(Model_path)
    
Checkpoint_path = Model_path + 'ckpt_weights/'
mkdir(Checkpoint_path)


# k-fold cross-validation
img, mask, dstmask, features = load_data_multi(train_frame_path, train_mask_path, train_maskdst_path, shape, band)
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
    
    train_y3 = features[train_list[i]]
    test_y3 = features[test_list[i]]
    
    
    #MODEL BUILD  multi_task
    if(network == 'unet'):
        m = model.get_unet_multitask(input_shape = (shape,shape,band))
    else:
        m = model.segnet(multi_task = True, input_shape = (shape,shape,band))
    
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

    #callback
    ckpt_path = Checkpoint_path + '%s/'%i
    mkdir(ckpt_path)
    
    weights_path = ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_binary_Mean_IOU:.2f}.hdf5'
    
    callbacks = get_callbacks(weights_path, Model_path, 5)

    if(aug):
    # data augmentation
        ratio = 0.18
        n = len(train_x)
        a = int(n*(1-ratio))
        b = n - a
        NO_OF_TRAINING_IMAGES = a
        NO_OF_VAL_IMAGES = b
        train_gen = MTgenerator(train_x[0:a], train_y[0:a], train_y2[0:a],  'train', BATCH_SIZE)
        val_gen = MTgenerator(train_x[a:], train_y[a:], train_y2[a:], 'val', BATCH_SIZE)

        history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data=val_gen,
                              validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                              shuffle = True,
                              callbacks=callbacks)
    else:
#         train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_noaug(train_x, train_y, 32, ratio = 0.18)
        history = m.fit(train_x, {'binary': train_y, 'distance': train_y2, 'classification':train_y3}, epochs=NO_OF_EPOCHS,
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

    score = m.evaluate(test_x, {'binary': test_y, 'distance': test_y2, 'classification':test_y3}, verbose=0)
    for j in range(14):
        print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
   

 #prediction
    results = m.predict(test_x)
    print(results[0].shape) # (n,128,128,5)

    #save image
    result_path = "/home/yifanc3/results/%s/%s/%s"%(date,Model_name,i)
    mkdir(result_path)

    print('result:', result_path)
    
    test_y = [test_y, test_y2, test_y3]
    save_result(train_frame_path, result_path, test_list[i], results, test_x, test_y, shape, multi_task=True)
    # saveFrame_256(save_frame_path, test_frame_path, X)
    print("======="*12, end="\n\n\n")
