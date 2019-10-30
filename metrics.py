import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
import cv2
from itertools import product


def Mean_IoU_cl(cl=2, shape=128):
    
    def Mean_IOU(y_true, y_pred):
        s = K.shape(y_true)

        # reshape such that w and h dim are multiplied together
        #revise
        y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, shape*shape, cl))
        y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, cl))
        # correctly classified
        clf_pred = K.one_hot( K.argmax(y_pred_reshaped), num_classes = s[-1])
        print(y_true_reshaped.dtype, y_pred_reshaped.dtype, clf_pred.dtype)
        print(np.shape(clf_pred), np.shape(y_true_reshaped), np.shape(y_pred_reshaped))
        equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

        # IoU for labeled class
    #     y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, 128*128, 2))
    #     y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, 128*128, 2))
    #     y_true_reshaped = K.cast(K.argmax(y_true_reshaped),dtype='float32')
    #     clf_pred = K.cast(K.argmax(y_pred_reshaped),dtype='float32')
    #     equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

        intersection = K.sum(equal_entries, axis=1)
        union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(clf_pred,axis=1)
        iou = intersection / (union_per_class - intersection)
        iou_mask = tf.is_finite(iou)
        iou_masked = tf.boolean_mask(iou,iou_mask)

        return K.mean( iou_masked )
    
    return Mean_IOU

    
    
def Mean_IOU_label(y_true, y_pred, shape=128):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    #MeanIoU all classes
#     y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, shape*shape, 2))
#     y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, 2))
#     # correctly classified
#     clf_pred = K.one_hot( K.argmax(y_pred_reshaped), num_classes = s[-1])
#     print(y_true_reshaped.dtype, y_pred_reshaped.dtype, clf_pred.dtype)
#     print(np.shape(clf_pred), np.shape(y_true_reshaped), np.shape(y_pred_reshaped))
#     equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

    # IoU for labeled class
    y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, 128*128, 2))
    y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, 128*128, 2))
    y_true_reshaped = K.cast(K.argmax(y_true_reshaped),dtype='float32')
    clf_pred = K.cast(K.argmax(y_pred_reshaped),dtype='float32')
    equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(clf_pred,axis=1)
    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou,iou_mask)

    return K.mean( iou_masked )



def precision_1(y_true, y_pred):
    """Precision metric.
    precision = TP/(TP + FP)
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    return TP/(TP + FP)

def precision_0(y_true, y_pred):
    """Precision metric.
    precision = TP/(TP + FP)
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    return TP/(TP + FP)


def recall_1(y_true, y_pred):
    """Recall metric.
    recall = TP/(TP+FN)
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP + FN)

def recall_0(y_true, y_pred):
    """Recall metric.
    recall = TP/(TP+FN)
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP + FN)


def f1score_1(y_true, y_pred):
    pre = precision_1(y_true, y_pred)
    rec = recall_1(y_true, y_pred)
    denominator = (pre + rec)
    numerator = (pre * rec)
    result = (numerator/denominator)*2
    return result

def f1score_0(y_true, y_pred):
    pre = precision_0(y_true, y_pred)
    rec = recall_0(y_true, y_pred)
    denominator = (pre + rec)
    numerator = (pre * rec)
    result = (numerator/denominator)*2
    return result

def per_pixel_acc(y_true, y_pred): # class1 and class0 actually the same
#     accuracy=(TP+TN)/(TP+TN+FP+FN)
    #class 1
    #y_pred = K.argmax(y_pred)
    y_pred = K.cast(K.greater(y_pred,0.5),'float32')
    #y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc0 = (TP)/(TP+FN)
    return acc0

def FP(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    if(FP+FN == 0):
        return 0
    return FP/(FP+FN)
    
def FN(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    if(FP+FN == 0):
        return 0
    return FN/(FP + FN)


# def iou_coef(y_true, y_pred, smooth=1):
#     """
#     IoU = (|X &amp; Y|)/ (|X or Y|)
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
#     return (intersection + smooth) / ( union + smooth)

# def iou_coef_loss(y_true, y_pred):
#     return -iou_coef(y_true, y_pred)
                  
                  
def iou_label(y_true, y_pred):
    ''' 
    calculate iou for label class
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''
    y_pred = K.cast(K.greater(y_pred,0.5),'float32')
#     y_pred = K.argmax(y_pred)
#     y_pred = K.greater(y_pred, 0.3)
#     y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP+FP+FN)


def iou_back(y_true, y_pred):
    ''' 
    calculate iou for background class
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''
    y_pred = 1-K.argmax(y_pred)
    y_true = 1-K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    return TP/(TP+FP+FN)

def accuracy(y_true, y_pred):
    '''calculate classification accuracy'''
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc