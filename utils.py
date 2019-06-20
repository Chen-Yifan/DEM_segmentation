import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from itertools import product

def saveResult(save_path, test_mask_path, results, flag_multi_class = False, num_class = 2):
    n = os.listdir(test_mask_path)
    shape = np.shape(results)
    print(shape)
    results = results.reshape(len(n),256,256,2)
    #results = results.astype('uint8')
    for i in range(shape[0]):
        img = np.argmax(results[i],axis = -1)
        img = np.squeeze(img)
        #cv2.imwrite(os.path.join(save_path,"%s_predict.png"%n[i][0:-4]),results[i])
        np.save(os.path.join(save_path,"%s_predict.npy"%n[i][0:-4]),img)
        
def saveMask_256(save_path, test_mask_path, test_mask):
    n = os.listdir(test_mask_path)
    shape = np.shape(test_mask)
    test_mask = test_mask.reshape(len(n),shape[1],shape[2],2)
    for i in range(len(n)):
        img = np.argmax(test_mask[i],axis = -1)
        np.save(os.path.join(save_path,"%s"%n[i]),img)
        
def saveFrame_256(save_path, test_frame_path, test_frame):
    n = os.listdir(test_frame_path)
    for i in range(len(n)):
        np.save(os.path.join(save_path,"%s"%n[i]),test_frame[i])
    
        

# plan B
def pixel_wise_loss(y_true, y_pred):
#     y_pred = K.argmax(y_pred)
#     y_true = K.argmax(y_true)

    y_true = tf.reshape(tensor=y_true, shape=(-1, 64*64, 2))
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, 64*64, 2))
    pos_weight = tf.constant([[1.0, 100.0]])# 150 won't change val_Mean_IOU while 500 makes IoU hard to exceed 0.60
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        y_pred,
        pos_weight,
        name=None
    )
   # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return K.mean(loss,axis=-1)

def soft_dice_loss(y_true, y_pred, smooth=1): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    y_true = tf.reshape(tensor=y_true, shape=(-1, 256* 256, 2))
#     y_true = np.array(y_true)
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, 256* 256, 2))
#     y_pred = np.array(y_pred)
#     # skip the batch and class axis for calculating Dice score
#     axes = tuple(range(1, len(y_pred.shape)-1))  # len = 4, tuple(range(1,3)) = (1,2)
#     print(np.shape(y_pred*y_true))
#     numerator = 2. * np.sum(y_pred * y_true, axes)
#     denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
#     print(np.shape(denominator))
#     return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch

    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


#debugging
def w_categorical_crossentropy(y_true, y_pred, weights=(np.array([[1,300],[1,300]]))):
    nb_cl = len(weights)
    y_true = tf.reshape(tensor=y_true, shape=(-1, 256* 256, 2))
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    print(K.shape(y_pred_max))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    print(y_pred_max_mat.shape)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        print(c_p, c_t)
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def softmax_with_entropy(y_true, y_pred):
    #y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    #y_pred = K.argmax(y_pred, axis=-1)
   #y_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))
    #y_true = K.argmax(y_true, axis=-1)
    print(K.int_shape(y_pred),K.int_shape(y_true))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return K.mean(loss, axis = -1)
    
# def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
#     y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
#     log_softmax = tf.nn.log_softmax(y_pred)

#     y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
#     unpacked = tf.unstack(y_true, axis=-1)
#     y_true = tf.stack(unpacked[:-1], axis=-1)

#     cross_entropy = -K.sum(y_true * log_softmax, axis=1)
#     cross_entropy_mean = K.mean(cross_entropy)

#     return cross_entropy_mean

def binary_crossentropy_with_logits(ground_truth, predictions):
    ground_truth = tf.reshape(tensor=ground_truth, shape=(-1, 256*256, 2))
#     predictions = tf.reshape(tensor=predictions, shape=(-1, 256*256, 2))
    print(predictions.shape, ground_truth.shape)
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)
def softmax_cross_entropy_with_logits(y_true, flat_logits):
    flat_labels = tf.reshape(tensor=y_true, shape=(-1, 256*256, 2))
#     flat_logits = tf.reshape(tensor=flat_logits, shape=(-1, 2))
#     flat_labels = K.argmax(flat_labels, axis=-1)
#     flat_logits = K.argmax(flat_logits, axis=-1)
#     flat_labels = tf.dtypes.cast(flat_labels,'float')
#     flat_logits = tf.dtypes.cast(flat_logits,'float') # no gradient
    print(flat_labels.shape, flat_logits.shape)
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
    return tf.reduce_sum(cross_entropies)
    


def Mean_IOU(y_true, y_pred):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    #revise
    y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, 64*64, 2))
    y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, 64*64, 2))
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
          
#     # reshape such that w and h dim are multiplied together
#     #revise

          
          
def dice_coef(y_true, y_pred, smooth = 0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


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
    return 2 * ((pre * rec) / (pre + rec))

def per_pixel_acc(y_true, y_pred):
#     accuracy=(TP+TN)/(TP+TN+FP+FN)
    #class 1
    y_pred = K.argmax(y_pred)
    y_true = K.argmax(y_true)
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
    
    