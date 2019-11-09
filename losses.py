import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
from keras.utils import to_categorical
from itertools import product

from scipy.spatial.distance import cdist
def dissimilarity_loss(y_true, y_pred):
    '''
    args:
        y_true y_pred are in the same dimension, in shape (dim,dim)
    return:
        dissimialirity loss
    '''
    #define evaluation metric, dissimilarity
    pred = y_pred
    gt = y_true
    a,b = np.where(pred==1)
    pred_idxs = np.column_stack((a,b))
    a,b = np.where(img==1)
    gt_idxs = np.column_stack((a,b)) # create a list with (row,column)
    loss = 0
    #loop through ones in pred find nearest ones in gt
    for pred_idx in pred_idxs:
        if gt[pred_idx[0],pred_idx[1]] == 1:
            continue
        elif len(gt_idxs)==0: # no corresponding 1 in gt array
            loss += 2*img.shape[0]
        else:
            dist_1 = cdist(np.array([[pred_idx[0],pred_idx[1]]]), gt_idxs,'cityblock')
            loss += dist_1.min()
            
    # loop through ones in gt find nearest ones in pred
    for gt_idx in gt_idxs:
        if gt[gt_idx[0],gt_idx[1]] == 1:
            continue
        elif len(pred_idxs)==0: # no corresponding 1 in gt array
            loss += 2*img.shape[0]
        else:
            dist_2 = cdist(np.array([[gt_idx[0],gt_idx[1]]]), pred_idxs,'cityblock')
            loss += dist_2.min()
     return loss
    
# plan B
def pixel_wise_loss(y_true, y_pred, shape=128):
#     y_pred = K.argmax(y_pred)
#     y_true = K.argmax(y_true)

    y_true = tf.reshape(tensor=y_true, shape=(-1, shape*shape, 2))
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, 2))
    pos_weight = tf.constant([1.0,300.0])# 150 won't change val_Mean_IOU while 500 makes IoU hard to exceed 0.60
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        y_pred,
        pos_weight,
        name=None
    )
   # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return K.mean(loss,axis=-1)

def weighted_categorical_crossentropy(weights): # after softmax
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def multi_weighted_loss_v2(y_true, y_pred, shape=128):
    y_true = tf.reshape(tensor=y_true, shape=(-1, shape*shape, 5))
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, 5))
    return tf.compat.v1.losses.softmax_cross_entropy(
        y_true,
        y_pred,
        weights=[1,20,50,100,200])

def multi_weighted_loss_v3(y_true, y_pred, shape=128):
    y_true = tf.reshape(tensor=y_true, shape=(-1, shape*shape, 5))
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, 5))
    class_weights = [1,20,50,100,200]
    class_weights = tf.multiply(y_true, class_weights)
    loss_crossentropy = tf.losses.softmax_cross_entropy(class_weights, y_pred)
    return tf.reduce_mean(loss_crossentropy)
          
def dice_coef(y_true, y_pred, smooth = 0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
