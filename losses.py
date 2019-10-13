import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
from keras.utils import to_categorical
from itertools import product

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

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss():
    return -dice_coef(y_true, y_pred)


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

def two_loss(weights):
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        dice_loss = dice_coef_loss(y_true, y_pred)
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss + dice_loss
    
    return loss

def multi_weighted_loss(y_true, y_pred, shape=128):
    y_true = tf.reshape(tensor=y_true, shape=(-1, shape*shape, 5))
    y_pred = tf.reshape(tensor=y_pred, shape=(-1, shape*shape, 5))
    
    weightsArray = [1,25,50,100,200]
    class_weight = tf.constant(weightsArray,dtype='float32')
    # Take the cost like normal
    error = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred,dim=1)
    print(error.shape)
    # Scale the cost by the class weights
    scaled_error = tf.math.multiply(error, class_weight)

    # Reduce
    return tf.reduce_mean(scaled_error)


# def sparse_cat_crossentropy_w(weight):
    
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.int32)
#         return tf.losses.sparse_softmax_cross_entropy(
#                 labels=y_true,
#                 logits=y_pred,
#                 weights=weight)
#     return loss


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

# def soft_dice_loss(y_true, y_pred, smooth=1): 
#     ''' 
#     Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
#     Assumes the `channels_last` format.
  
#     # Arguments
#         y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
#         y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
#         epsilon: Used for numerical stability to avoid divide by zero errors
    
#     # References
#         V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
#         https://arxiv.org/abs/1606.04797
#         More details on Dice loss formulation 
#         https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
#         Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
#     '''
#     y_true = tf.reshape(tensor=y_true, shape=(-1, 256* 256, 2))
# #     y_true = np.array(y_true)
#     y_pred = tf.reshape(tensor=y_pred, shape=(-1, 256* 256, 2))
# #     y_pred = np.array(y_pred)
# #     # skip the batch and class axis for calculating Dice score
# #     axes = tuple(range(1, len(y_pred.shape)-1))  # len = 4, tuple(range(1,3)) = (1,2)
# #     print(np.shape(y_pred*y_true))
# #     numerator = 2. * np.sum(y_pred * y_true, axes)
# #     denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
# #     print(np.shape(denominator))
# #     return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch

#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


#debugging
# def w_categorical_crossentropy(y_true, y_pred, weights=(np.array([[1,300],[1,300]]))):
#     nb_cl = len(weights)
#     y_true = tf.reshape(tensor=y_true, shape=(-1, 256* 256, 2))
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#     print(K.shape(y_pred_max))
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     print(y_pred_max_mat.shape)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         print(c_p, c_t)
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask


# def softmax_with_entropy(y_true, y_pred):
#     #y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
#     #y_pred = K.argmax(y_pred, axis=-1)
#    #y_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))
#     #y_true = K.argmax(y_true, axis=-1)
#     print(K.int_shape(y_pred),K.int_shape(y_true))
#     loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     return K.mean(loss, axis = -1)
    
# def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
#     y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
#     log_softmax = tf.nn.log_softmax(y_pred)

#     y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
#     unpacked = tf.unstack(y_true, axis=-1)
#     y_true = tf.stack(unpacked[:-1], axis=-1)

#     cross_entropy = -K.sum(y_true * log_softmax, axis=1)
#     cross_entropy_mean = K.mean(cross_entropy)

#     return cross_entropy_mean

# def binary_crossentropy_with_logits(ground_truth, predictions):
#     ground_truth = tf.reshape(tensor=ground_truth, shape=(-1, 256*256, 2))
# #     predictions = tf.reshape(tensor=predictions, shape=(-1, 256*256, 2))
#     print(predictions.shape, ground_truth.shape)
#     return K.mean(K.binary_crossentropy(ground_truth,
#                                         predictions,
#                                         from_logits=True),
#                   axis=-1)
# def softmax_cross_entropy_with_logits(y_true, flat_logits):
#     flat_labels = tf.reshape(tensor=y_true, shape=(-1, 256*256, 2))
# #     flat_logits = tf.reshape(tensor=flat_logits, shape=(-1, 2))
# #     flat_labels = K.argmax(flat_labels, axis=-1)
# #     flat_logits = K.argmax(flat_logits, axis=-1)
# #     flat_labels = tf.dtypes.cast(flat_labels,'float')
# #     flat_logits = tf.dtypes.cast(flat_logits,'float') # no gradient
#     print(flat_labels.shape, flat_logits.shape)
#     cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
#     return tf.reduce_sum(cross_entropies)
    

          
          
def dice_coef(y_true, y_pred, smooth = 0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
