import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
import cv2
from keras.utils import to_categorical

one = [255,255,255]
two = [0,0,0]

COLOR_DICT = np.array([two, one])

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


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
    test_mask = test_mask.reshape(len(n),256,256,2)
    for i in range(len(n)):
#         test_mask_0 = np.load(test_mask_path+'/'+n[i]) # 1.0 or 2.0
# #         #change 2--0
#         test_mask = np.where(test_mask_0==2, 0, test_mask_0)
#         # resize to 256, 256, 1
#         #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
#         test_mask = cv2.resize(test_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype('uint8')
#         #test_mask = test_mask.reshape(256, 256, 1)
#         #test_mask = to_categorical(test_mask, 2)
# #        img = np.zeros((256, 256, 1), dtype=np.uint8)
# #         img[:,:,0] = np.squeeze(test_mask)
# #         img[:,:,1] = np.squeeze(1-test_mask)
        img = np.argmax(test_mask[i],axis = -1)
        np.save(os.path.join(save_path,"%s"%n[i]),img)
        
def saveFrame_256(save_path, test_frame_path, test_frame):
    n = os.listdir(test_frame_path)
    for i in range(len(n)):
        np.save(os.path.join(save_path,"%s"%n[i]),test_frame[i])
    
    
        
# 1--- black1 . 2 --- white0
def visMaskImage(save_path, test_mask_path):
    pass

# plan B
def pixel_wise_loss(y_true, y_pred):
#     y_pred = K.argmax(y_pred)
#     y_true = K.argmax(y_true)
    print(K.int_shape(y_pred))
    y_true = tf.reshape(tensor=y_true, shape=(-1, 256*256, 2))
    pos_weight = tf.constant([[1.0, 500.0]])# 150 won't change val_Mean_IOU while 500 makes IoU hard to exceed 0.60
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        y_pred,
        pos_weight,
        name=None
    )
   # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return K.mean(loss,axis=-1)

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
    

# from https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
# def Mean_IOU(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     iou = []
#     true_pixels = K.argmax(y_true, axis=-1)
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     void_labels = K.equal(K.sum(y_true, axis=-1), 0)
#     for i in range(0, nb_classes): # exclude first label (background) and last label (void)
#         true_labels = K.equal(true_pixels, i) & ~void_labels
#         pred_labels = K.equal(pred_pixels, i) & ~void_labels
#         inter = tf.to_int32(true_labels & pred_labels)
#         union = tf.to_int32(true_labels | pred_labels)
#         legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
#         ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
#         iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
#     iou = tf.stack(iou)
#     legal_labels = ~tf.debugging.is_nan(iou)
#     iou = tf.gather(iou, indices=tf.where(legal_labels))
#     return K.mean(iou)

def Mean_IOU(y_true, y_pred):
    s = K.shape(y_true)
    print(s)

    # reshape such that w and h dim are multiplied together
    #revise
    y_true_reshaped = tf.reshape(tensor=y_true, shape=(-1, 256*256, 2))
    print(y_true.shape)
    y_pred_reshaped = tf.reshape(tensor=y_pred, shape=(-1, 256*256, 2))
    print(y_pred.shape)
    # correctly classified
    clf_pred = K.one_hot( K.argmax(y_pred_reshaped), num_classes = s[-1])
    equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(y_pred_reshaped,axis=1)

    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou,iou_mask)

    return K.mean( iou_masked ) 
