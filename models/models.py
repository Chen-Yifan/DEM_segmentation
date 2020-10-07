import os
import numpy as np
from keras import backend as K
from skimage.io import imsave
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from metrics import *
from losses import *
from keras.regularizers import l2
import tensorflow as tf
import lovasz_losses_tf as L
from tensorflow.keras.metrics import MeanIoU, Precision, Recall, BinaryAccuracy

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def unet_shirui(channels=1, lmbda=1e-6, drop=0.45, init=None, n_filters=32, output_mode='sigmoid', 
                learn_rate=1e-4):
    
    inputs = Input((128,128,channels))
    conv1 = Conv2D(n_filters, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(n_filters*2, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    
    conv3 = Conv2D(n_filters*4, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(pool2)
    batch3 = BatchNormalization()(conv3)
    conv4 = Conv2D(n_filters*4, (3, 3),kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(batch3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)
    
    conv5 = Conv2D(n_filters*4, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(pool4)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    up6 = Dropout(drop)(up6)
    conv6 = Conv2D(n_filters*2, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(up6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2], axis=3)
    up7 = Dropout(drop)(up7)
    conv7 = Conv2D(n_filters, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation='relu', padding='same')(up7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1], axis=3)
    up8 = Dropout(drop)(up8)
    conv8 = Conv2D(n_filters, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init,activation='relu', padding='same')(up8)
    conv9 = Conv2D(n_filters, (3, 3), kernel_regularizer=l2(lmbda), kernel_initializer=init,activation='relu', padding='same')(conv8)

    conv10 = Conv2D(1, (1, 1), kernel_regularizer=l2(lmbda), kernel_initializer=init, activation=output_mode)(conv9) 
    
    conv10 = Reshape((128, 128, 1))(conv10)
    model = Model(inputs = inputs, outputs = conv10)
    
    optimizer = Adam(lr=learn_rate)
    
    #optimizer = Adadelta()
    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), per_pixel_acc(threshold=0), accuracy(threshold=0)], optimizer=optimizer)
    elif output_mode == 'sigmoid':
        model.compile(loss='binary_crossentropy', metrics=[iou_label(),per_pixel_acc(),accuracy()], optimizer=optimizer)
    else:
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0),per_pixel_acc(threshold=0),accuracy(threshold=0)], optimizer=optimizer)
        
    model.summary()

    return model


def unet_rgl(channels=1, lr=1e-4, n_filters=64, output_mode='sigmoid', lmbda=1e-6):
    inputs = Input((128, 128, channels))
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)   
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5) 
    up6 = Conv2D(n_filters*8, 2, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv6)   
    up7 = Conv2D(n_filters*4, 2, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv7)   
    up8 = Conv2D(n_filters*2, 2, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv8)   
    up9 = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_regularizer=l2(lmbda),
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  
    model = Model(inputs, conv10)  
    optimizer = Adam(lr)

    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), per_pixel_acc(
            threshold=0), accuracy(threshold=0)], optimizer=optimizer)
    elif output_mode == 'sigmoid':
        model.compile(loss=wbce, metrics=[
                      iou_label(), per_pixel_acc(), accuracy()], optimizer=optimizer)
    else: # None
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), per_pixel_acc(
            threshold=0), accuracy(threshold=0)], optimizer=optimizer)


    # model.compile(loss='binary_crossentropy',metrics=[iou_label(),per_pixel_acc(),accuracy()], optimizer=Adam(lr=1e-4))
    model.summary()

    return model

def unet(channels=1, lr=1e-4, n_filters=64, output_mode='sigmoid'):
    inputs = Input((128, 128, channels))
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)   
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5) 
    up6 = Conv2D(n_filters*8, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)   
    up7 = Conv2D(n_filters*4, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)   
    up8 = Conv2D(n_filters*2, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)   
    up9 = Conv2D(n_filters, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  
    model = Model(inputs, conv10)   
    optimizer = Adam(lr)

    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), per_pixel_acc(
            threshold=0), accuracy(threshold=0)], optimizer=optimizer)
    elif output_mode == 'sigmoid':
        model.compile(loss=wbce, metrics=[
                      iou_label(), per_pixel_acc(), accuracy()], optimizer=optimizer)
    else:  # None
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), per_pixel_acc(
            threshold=0), accuracy(threshold=0)], optimizer=optimizer)

    model.summary()

    return model


