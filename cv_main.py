import os
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint,CSVLogger, EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD


import tensorflow as tf
from models import models,deeplab
from metrics import *
from losses import *
from helper import *
from util.util import *
from util.summarize import *

from options.train_options import TrainOptions

#Options
opt = TrainOptions().parse()

BATCH_SIZE = opt.batch_size
NO_OF_EPOCHS = opt.epochs
shape = opt.input_shape
inc = opt.input_channel
aug = opt.augmentation # to decide if shuffle
Model_path = opt.Model_path # os.path.join(opt.checkpoints_dir, opt.date, opt.name)
Result_path = opt.Result_path
model = opt.model
k = opt.k
frame_path = opt.frame_path
mask_path = opt.mask_path
date = opt.date
Model_name = opt.name
loss_function = opt.loss_function
weight = opt.weight

mkdir(Model_path)
Checkpoint_path = Model_path + '/ckpt_weights/'
mkdir(Checkpoint_path)


# k-fold cross-validation
img, mask = load_data(frame_path, mask_path, shape, inc)
train_list, test_list = k_fold(len(img), k = k)
print(len(train_list), len(test_list))

model_history = [] 

for i in range(1,k):
    print('====The %s Fold===='%i)
    #shuffle the index
#     random.shuffle(train_list[i])
#     random.shuffle(test_list[i])
    
    train_x = img[train_list[i]]
    train_y = mask[train_list[i]]
    test_x = img[test_list[i]]
    test_y = mask[test_list[i]]
    
    '''Define_model'''
    
    input_shape = (shape,shape,inc)
    if(model == 'unet'):
        m = models.unet(input_shape=input_shape)
    elif(model == 'segnet'):
        m = models.segnet(input_shape=input_shape)
    elif(model == 'resnet'):
        m = models.resnet(input_shape=input_shape)
    else:
        m = deeplab.DeeplabV2(input_shape=input_shape)
        
    # optimizer    
    if(opt.optimizer==2):
        optimizer = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    else:
        optimizer = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    
    weights = np.array([1.0,weight])
    
    if(loss_function=='single'):
        loss = weighted_categorical_crossentropy(weights) 
    else:
        loss = two_loss(weights)
    
    Mean_IOU = Mean_IoU_cl(cl=2)
    m.compile( optimizer = optimizer, loss = loss, metrics = [per_pixel_acc, Mean_IOU, Mean_IOU_label, precision, recall, f1score])

    #callback
    ckpt_path = Checkpoint_path + '%s/'%i
    mkdir(ckpt_path)
    weights_path = ckpt_path +'weights.{epoch:02d}-{val_loss:.2f}-{val_Mean_IOU:.2f}-{val_Mean_IOU_label:.2f}.hdf5'
    
    callbacks = get_callbacks(i, opt.optimizer, weights_path, Model_path, 5)
    
    if(aug):
        print('aug==1')
 # data augmentation
        train_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_aug(train_x, train_y, 32, ratio = 0.18)
        val_img = train_x[NO_OF_TRAINING_IMAGES:]
        val_mask = train_y[NO_OF_TRAINING_IMAGES:]
        history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data=(val_img,val_mask),
                              validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                              shuffle = True,
                              callbacks=callbacks)
    else:
        print('aug==0')
#         train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_noaug(train_x, train_y, 32, ratio = 0.18)
        history = m.fit(train_x, train_y, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                         verbose=1, validation_split=0.18, shuffle = True)
    
    model_history.append(history)
    
    '''Save model'''
    model_json = m.to_json()
    with open(os.path.join(Model_path,"model%s.json" %i), "w") as json_file:
        json_file.write(model_json)

    # convert the history.history dict to a pandas DataFrame: 
    hist_df = pd.DataFrame(history.history) 
    # save to json:  
    hist_json_file = 'history_%s.json'%i 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
        
    # serialize weights to HDF5
    print("Saved model to disk")
    m.save(os.path.join(Model_path,'model%s.h5' %i))
    
    '''Evaluate Model'''
    print('======Start Testing======')
    score = m.evaluate(test_x, test_y, verbose=0)

    message = ''
    for j in range(7):
        print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
        message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)
        
    output_file = os.path.join(Model_path, 'output_%s'%i)
    with open(output_file, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
    '''Save Result'''
    results = m.predict(test_x)
    new_r = np.argmax(results,axis=-1)

    #save image
    result_path = os.path.join(Result_path, '%s'%i)
    mkdir(result_path)
    print('result:', result_path)
    
    save_result(frame_path, result_path, test_list[i], results, test_x, test_y, shape)
    # saveFrame_256(save_frame_path, test_frame_path, X)
    print("======="*12, end="\n\n\n")
    
    
    '''Summarize Model '''
    summarize_performance(history, Model_path)
