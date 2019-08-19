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


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = K.expand_dims(K.cast(K.argmax(y_pred),dtype='float32'), axis=-1)
    y_true = K.expand_dims(K.cast(K.argmax(y_true),dtype='float32'), axis=-1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = K.expand_dims(K.cast(K.argmax(y_pred),dtype='float32'), axis=-1)
    y_true = K.expand_dims(K.cast(K.argmax(y_true),dtype='float32'), axis=-1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    denominator = (pre + rec)
    numerator = (pre * rec)
    result = (numerator/denominator)*2
    return result

def per_pixel_acc(y_true, y_pred):
#     accuracy=(TP+TN)/(TP+TN+FP+FN)
    #class 1
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
   # TP = tf.compat.v2.math.count_nonzero(y_pred * y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc0 = (TP+TN)/(TP+TN+FP+FN)
    #class 0
    y_pred = 1-y_pred
    y_true = 1-y_true
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc1 = (TP+TN)/(TP+TN+FP+FN)    
    return (acc0 + acc1)/2

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

# def TP(y_true, y_pred):
#     y_pred = K.argmax(y_pred)
#     y_true = K.argmax(y_true)
#     TP = tf.math.count_nonzero(y_pred * y_true)
    
#     return TP

# def TN(y_true, y_pred):
#     y_pred = K.argmax(y_pred)
#     y_true = K.argmax(y_true)
#     TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
#     TP = tf.math.count_nonzero(y_pred * y_true)
#     return TN / (TN + TP)

def accuracy(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
    FP = tf.math.count_nonzero(y_pred*(1-y_true))
    FN = tf.math.count_nonzero((1-y_pred)*y_true)
    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc