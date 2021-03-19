import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
from keras.utils import to_categorical
from itertools import product

from scipy.spatial.distance import cdist

def wbce( y_true, y_pred, weight1=0.7, weight0=0.3) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)
    
ALPHA = 0.7
BETA = 0.3
GAMMA = 1

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
           
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1 - Tversky), gamma)
    
    return FocalTversky
        
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

def sparse_softmax_cce(y_true, y_pred):
    if len(y_true.get_shape()) == 4:
        y_true = K.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, 'uint8')
    return tf.keras.backend.sparse_categorical_crossentropy(y_true,y_pred)

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

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

