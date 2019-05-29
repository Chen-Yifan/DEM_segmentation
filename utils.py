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
    results = results.reshape(len(n),256,256,2)
    for i in range(shape[0]):
        img = results[i]
        img = np.squeeze(img)
        #cv2.imwrite(os.path.join(save_path,"%s_predict.png"%n[i][0:-4]),results[i])
        np.save(os.path.join(save_path,"%s_predict.npy"%n[i][0:-4]),img)
        
def saveMask_256(save_path, test_mask_path):
    n = os.listdir(test_mask_path)
    for i in range(len(n)):
        test_mask_0 = np.load(test_mask_path+'/'+n[i]) # 1.0 or 2.0
#         #change 2--0
        test_mask = np.where(test_mask_0==2, 0, test_mask_0)
        # resize to 256, 256, 1
        #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
        test_mask = cv2.resize(test_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype('uint8')
        #test_mask = test_mask.reshape(256, 256, 1)
        #test_mask = to_categorical(test_mask, 2)
        img = np.zeros((256, 256, 2), dtype=np.uint8)
        img[:,:,0] = np.squeeze(test_mask)
        img[:,:,1] = np.squeeze(1-test_mask)
        np.save(os.path.join(save_path,"%s"%n[i]),img)
    
        
# 1--- black1 . 2 --- white0
def visMaskImage(save_path, test_mask_path):
    pass

# plan B
def pixel_wise_loss(y_true, y_pred):
    pos_weight = tf.constant([[1.0, 2.0]])
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        y_pred,
        pos_weight,
        name=None
    )
    return K.mean(loss,axis=-1)

# from https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

