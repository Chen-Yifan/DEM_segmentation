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


# customized class for SegNet
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        input_shape[0],
                        input_shape[1]*self.size[0],
                        input_shape[2]*self.size[1],
                        input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                    [[input_shape[0]], [1], [1], [1]],
                    axis=0)
            batch_range = K.reshape(
                    K.tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )

def unet_shirui(channels=1, lmbda=1e-6, drop=0.45, init=None, n_filters=32, optloss='bce', output_mode='sigmoid', 
                learn_rate=1e-4): 
    # Shirui's implementation during Hackthon
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
    
    opt = Adam(lr=learn_rate)
    
    #optimizer = Adadelta()
        
    if optloss == 'cce':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), dice_coefficient(
            threshold=0), accuracy(threshold=0)], optimizer=opt)
    elif optloss=='bce':
        model.compile(loss='binary_crossentropy', metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    elif optloss=='wbce':
        model.compile(loss=wbce, metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    elif optloss=='T':
        model.compile(loss=FocalTverskyLoss, metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    else:  # L = 'L'
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), dice_coefficient(threshold=0), accuracy(threshold=0)], optimizer=Adam(lr=3e-4))


    model.summary()

    return model


def unet_rgl(channels=1, learn_rate=1e-4, n_filters=64, pretrained_weights=None, output_mode='sigmoid', lmbda=1e-6): 
    # unet with kernel weight regularizer
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
    conv10 = Conv2D(1, 1, activation=output_mode, kernel_regularizer=l2(lmbda))(conv9)  
    conv10 = Reshape((128, 128, 1))(conv10)
    
    model = Model(inputs, conv10)  
    optimizer = Adam(lr=learn_rate)

    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
        
    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), dice_coefficient(
            threshold=0), accuracy(threshold=0)], optimizer=optimizer)
    elif output_mode == 'sigmoid':
        model.compile(loss='binary_crossentropy', metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=optimizer)
    else: # None
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), dice_coefficient(threshold=0), accuracy(threshold=0)], optimizer=optimizer)


    # model.compile(loss='binary_crossentropy',metrics=[iou_label(),per_pixel_acc(),accuracy()], optimizer=Adam(lr=1e-4))
    model.summary()

    return model
    
    
def conv2d(n, activation_opt, inputs, kernel_init='he_normal'):
    return Conv2D(n, 3, activation = activation_opt, padding = 'same', kernel_initializer = kernel_init)(inputs)
    
def unet(channels=1, learn_rate=1e-4, n_filters=64, activation_opt='relu', optloss='bce', output_mode='sigmoid',pretrained_weights=None):
    inputs = Input((128, 128, channels))
    conv1 = conv2d(n_filters, activation_opt, inputs)
    conv1 = conv2d(n_filters, activation_opt, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv2d(n_filters*2, activation_opt, pool1)
    conv2 = conv2d(n_filters*2, activation_opt, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv2d(n_filters*4, activation_opt, pool2)
    conv3 = conv2d(n_filters*4, activation_opt, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv2d(n_filters*8, activation_opt, pool3)
    conv4 = conv2d(n_filters*8, activation_opt, conv4)
    # drop4 = Dropout(0.45)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d(n_filters*16, activation_opt, pool4)
    conv5 = conv2d(n_filters*16, activation_opt, conv5)
    # drop5 = Dropout(0.45)(conv5)

    up6 = conv2d(n_filters*8, activation_opt, UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    drop6 = Dropout(0.45)(merge6)
    conv6 = conv2d(n_filters*8, activation_opt, drop6)
    conv6 = conv2d(n_filters*8, activation_opt, conv6)

    up7 = conv2d(n_filters*4, activation_opt, UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    drop7 = Dropout(0.45)(merge7)
    conv7 = conv2d(n_filters*4, activation_opt, drop7)
    conv7 = conv2d(n_filters*4, activation_opt, conv7)

    up8 = conv2d(n_filters*2, activation_opt, UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    drop8 = Dropout(0.45)(merge8)
    conv8 = conv2d(n_filters*2, activation_opt, drop8)
    conv8 = conv2d(n_filters*2, activation_opt, conv8)

    up9 = conv2d(n_filters, activation_opt, UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = conv2d(n_filters, activation_opt, merge9)
    conv9 = conv2d(n_filters, activation_opt, conv9)
    conv9 = conv2d(2, activation_opt, conv9)
    
    conv10 = Conv2D(1, 1, activation = output_mode)(conv9)

    model = Model(input = inputs, output = conv10)

    opt = Adam(lr = learn_rate)
    # opt = Adadelta(learn_rate)

    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
        
    if optloss == 'cce': # sparse_softmax_cce
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), dice_coefficient(
            threshold=0), accuracy(threshold=0)], optimizer=opt)
    elif optloss=='bce': # binary_crossentropy
        model.compile(loss='binary_crossentropy', metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    elif optloss=='wbce': # weighted binary crossentropy loss
        model.compile(loss=wbce, metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    elif optloss=='T': # FocalTverskyLoss
        model.compile(loss=FocalTverskyLoss, metrics=[
                      iou_label(), dice_coefficient(), accuracy()], optimizer=opt)
    else:  # L = 'L' # lovasz_loss
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), dice_coefficient(threshold=0), accuracy(threshold=0)], optimizer=Adam(lr=3e-4))

    model.summary()

    return model


def unet2(channels=1, lr=1e-4, n_filters=64, output_mode='sigmoid'):
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
    optimizer = Adam(lr=1e-4)

    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(threshold=0), accuracy(threshold=0)], optimizer=optimizer)
    elif output_mode == 'sigmoid':
        model.compile(loss='binary_crossentropy', metrics=[
                      iou_label(), accuracy()], optimizer=optimizer)
    else:  # None
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0), accuracy(threshold=0)], optimizer=optimizer)

    model.summary()

    return model
    

def segnet(
        n_classes,
        input_shape=(128,128,1),
        output_mode="sigmoid",
        kernel=3,
        pool_size=(2, 2)):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_classes, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    conv_26 = Reshape((128, 128, n_classes))(conv_26)
    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    
    optimizer = Adam(lr=3e-4)
    # optimizer = Adadelta()
    if output_mode == 'softmax':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label,per_pixel_acc,'accuracy'], optimizer=optimizer)
    else:
        model.compile(loss='binary_crossentropy',metrics=[iou_label,per_pixel_acc,'accuracy'], optimizer=optimizer)
        
    model.summary()

    return model
