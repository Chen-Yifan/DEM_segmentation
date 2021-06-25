import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
import cv2
from itertools import product
from skimage.morphology import skeletonize

def centerline_acc(y_true, y_pred):
    """
    acc = (#( y_true_center & y_pred ) / #y_true_center + #( y_pred_center & y_true ) / #y_pred_center) / 2
    Average of ( ratio of right prediction on centerline + ratio of predicted centerline in the groundtruth buffer)
    """
    smooth = 0.01
    y_pred = (y_pred >= 0.5).astype('uint8')
    y_true = y_true.astype('uint8')
    n = len(y_true)
    acc = 0
    for i in range(n):
        y_pred_curr = np.squeeze(y_pred[i])
        y_true_curr = np.squeeze(y_true[i])
        
        y_true_center = skeletonize(y_true_curr).astype('uint8')     
        tmp = np.sum(y_true_center&y_pred_curr)/(np.sum(y_true_center) + smooth)

        y_pred_center = skeletonize(y_pred_curr).astype('uint8')
        tmp2 = np.sum(y_pred_center&y_true_curr)/(np.sum(y_pred_center) + smooth)

#         if(np.sum(y_true_center)<10 or np.sum(y_pred_center)<10): # if there is too little features in an image, ignore it
#             n-=1
#             continue
            
        acc += (tmp + tmp2)/2

    return acc/n

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


def dice_coefficient(threshold=0.5): # class1 and class0 actually the same
    def dice(y_true, y_pred):
    #     accuracy=(TP+TN)/(TP+TN+FP+FN)
        #class 1
        if(y_pred.shape[-1]==2): # one-hot
            y_pred = K.cast(K.argmax(y_pred,axis=-1),'uint8')
        elif(y_pred.shape[-1]==1):
            y_pred = K.cast(K.greater(K.squeeze(y_pred,axis=-1),threshold),'uint8')
        y_true = K.cast(K.squeeze(y_true,axis=-1),'uint8')

        TP = tf.math.count_nonzero(y_pred * y_true)
        TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
        FP = tf.math.count_nonzero(y_pred*(1-y_true))
        FN = tf.math.count_nonzero((1-y_pred)*y_true)
        acc1 = (2*TP)/(2*TP+FN+FP)
        return acc1
    return dice
                  
                  
def iou_label(threshold=0.5):
    def iou(y_true, y_pred):
        ''' 
        calculate iou for label class
        IOU = true_positive / (true_positive + false_positive + false_negative)
        '''
        print(y_true.shape,y_pred.shape)
        if(y_pred.shape[-1]==2): # one-hot
            y_pred = K.cast(K.argmax(y_pred,axis=-1),'uint8')
        elif(y_pred.shape[-1]==1):
            y_pred = K.cast(K.greater(K.squeeze(y_pred,axis=-1),threshold),'uint8')
        y_true = K.cast(K.squeeze(y_true,axis=-1),'uint8')
        TP = tf.math.count_nonzero(y_pred * y_true)
        TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
        FP = tf.math.count_nonzero(y_pred*(1-y_true))
        FN = tf.math.count_nonzero((1-y_pred)*y_true)
        return TP/(TP+FP+FN)
    return iou


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

def accuracy(threshold=0.5):
    def acc(y_true, y_pred):
        '''calculate classification accuracy'''
        if(y_pred.shape[-1]==2): # one-hot
            y_pred = K.cast(K.argmax(y_pred,axis=-1),'uint8')
        elif(y_pred.shape[-1]==1):
            y_pred = K.cast(K.greater(K.squeeze(y_pred,axis=-1), threshold),'uint8')
        y_true = K.cast(K.squeeze(y_true,axis=-1),'uint8')

        TP = tf.math.count_nonzero(y_pred * y_true)
        TN = tf.math.count_nonzero((1-y_pred)*(1-y_true))
        FP = tf.math.count_nonzero(y_pred*(1-y_true))
        FN = tf.math.count_nonzero((1-y_pred)*y_true)
        result = (TP+TN)/(TP+TN+FP+FN)
        return result
    return acc
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

    
