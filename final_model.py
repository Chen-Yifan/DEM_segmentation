# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:59 2019

@author: Shiru
"""

from models.models import unet
from metrics import iou_label
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split
from keras.metrics import binary_accuracy
from keras.models import Model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

from keras import __version__ as keras_version
k2 = True if keras_version[0] == '2' else False
from keras.layers import BatchNormalization
if not k2:
    from keras.layers import merge, Input
    from keras.layers.convolutional import (Convolution2D, MaxPooling2D, UpSampling2D)

else:
    from keras.layers import Concatenate, Input
    from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                            UpSampling2D)

    def merge(layers, mode=None, concat_axis=None):
        """Wrapper for Keras 2's Concatenate class (`mode` is discarded)."""
        return Concatenate(axis=concat_axis)(list(layers))

    def Convolution2D(n_filters, FL, FLredundant, activation=None,
                      init=None, W_regularizer=None, border_mode=None):
        """Wrapper for Keras 2's Conv2D class."""
        return Conv2D(n_filters, (FL,FL), activation=activation,
                      kernel_initializer=init,
                      kernel_regularizer=W_regularizer,
                      padding=border_mode)


def read_input(mypath):
    
    #mypath = 'C:/Users/Shiru/Desktop/dem/all_frames_DEM'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    nfiles = len(onlyfiles)
    os.chdir(mypath)
    input_data = np.zeros((nfiles,128,128))
    for i in range(nfiles):
        input_data[i,:,:]=np.load(onlyfiles[i])

        dx = np.gradient(input_data[i,:,:])[0]
        dy = np.gradient(input_data[i,:,:])[1] 
        input_data[i,:,:] = np.sqrt((dx*dx)+(dy*dy))		
    return input_data


def read_label(mypath):
    
    #mypath = 'C:\\Users\\Shiru\\Desktop\\dem\\all_masks_5m6b'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    nfiles = len(onlyfiles)
    os.chdir(mypath)
    label_data = np.zeros((nfiles,128,128))
    for i in range(nfiles):
        label_data[i,:,:]=np.load(onlyfiles[i])
        
    return label_data

def is_feature_present(input_array):
    # for j in range(128):
        # for k in range(128):
            # if(input_array[j][k] > 0.5):
                # return True
    return (np.sum(input_array)>0)

def select_feature_images(folder_input,folder_label):
    
    data_input = np.zeros((6000, 128, 128))
    data_label = np.zeros((6000, 128, 128))
    
    i = 0

    for filename in listdir(folder_label):
        input_array = np.load(folder_input + "/" + filename)
        label_array = np.load(folder_label + "/" + filename)

        if(is_feature_present(label_array)):
            data_input[i] = input_array
            data_label[i] = label_array
            i += 1

    data_input = data_input[:i]
    data_label = data_label[:i]
    return data_input, data_label

def select_feature_images_gradient(folder_input,folder_label):
    
    data_input = np.zeros((6000, 128, 128))
    data_label = np.zeros((6000, 128, 128))
    
    i = 0

    for filename in listdir(folder_label):
        input_array = np.load(folder_input + "/" + filename)
        label_array = np.load(folder_label + "/" + filename)

        if(is_feature_present(label_array)):
            dx = np.gradient(input_array[i,:,:])[0]
            dy = np.gradient(input_array[i,:,:])[1] 
            data_input[i,:,:] = np.sqrt((dx*dx)+(dy*dy))
            
            data_label[i] = label_array
            i += 1

    data_input = data_input[:i]
    data_label = data_label[:i]
    return data_input, data_label

def visualize(real_data, predict_data, predicted_data, idx):
    
    f = plt.figure(figsize = (10,5))
    
    f.add_subplot(1,3,1)
    cs = plt.imshow(real_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('real')
    
    f.add_subplot(1,3,2)
    cs = plt.imshow(predict_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('predicted')
    
    # predicted_data = np.zeros((predict_data.shape[0], predict_data.shape[1], predict_data.shape[2]))
    # f.add_subplot(1,3,3)
    # for i in range(predict_data.shape[0]):
        # for j in range(predict_data.shape[1]):
            # for k in range(predict_data.shape[2]):
                # if (predict_data[i,j,k]>=0.3):
                    # predicted_data[i,j,k] =1
                # else:
                    # predicted_data[i,j,k] =0
    f.add_subplot(1,3,3)
    cs = plt.imshow(predicted_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('pred_processed')	
    plt.show()
    plt.savefig(MP['save_dir']+'/mask_comparison_idx%d.png'%(idx))
    plt.close()

def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.
    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.
    Yields
    ------
    Manipulated images and targets.
        
    """
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
																npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)

def build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network. 
    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter. 
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type. see https://keras.io/initializers/ for all the options
        use he_normal for relu activation function
    n_filters : int
        Number of filters in each layer.
    Returns
    -------
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    # a1 = BatchNormalization()(a1)
    # a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       # W_regularizer=l2(lmbda), border_mode='same')(a1)

    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)
    a1P = BatchNormalization()(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    # a2 = BatchNormalization()(a2)
    # a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       # W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a2P = BatchNormalization()(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = BatchNormalization()(a3)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)
    u = BatchNormalization()(a3P)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    # u = BatchNormalization()(u)
    # u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      # W_regularizer=l2(lmbda), border_mode='same')(u)
    u = UpSampling2D((2, 2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    # u = BatchNormalization()(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      # W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    # u = BatchNormalization()(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    # u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      # W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    # u = BatchNormalization()(u)	
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    # u = BatchNormalization()(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    # u = BatchNormalization()(u)	
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    
    model = Model(inputs=img_input, outputs=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', metrics=['accuracy',iou_label], optimizer=optimizer)
    print(model.summary())

    return model

def train_and_test_model(Data, MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model. 
    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = MP['filter_length']
    learn_rate = MP['lr']
    n_filters = MP['n_filters']
    init = MP['init']
    lmbda = MP['lambda']
    drop = MP['dropout']

    # Build model
    model = build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters)
#     model = unet()
    # Main loop
    n_samples = MP['n_train']
#     for nb in range(nb_epoch):
    model.fit_generator(
            custom_image_generator(Data['train'][0], Data['train'][1],
                                   batch_size=bs),
            steps_per_epoch=n_samples/bs, epochs=nb_epoch, verbose=1,
            validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
#             validation_data=custom_image_generator(Data['dev'][0],
#                                                    Data['dev'][1],
#                                                    batch_size=bs),
            validation_steps=n_samples,
#             callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)]
    )

        #get_metrics(Data['train'], Craters['train'], dim, model)

#     if MP['save_models'] == 1:
#         model.save(MP['save_dir'])

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%d, 
    n_epoch=%d, n_train=%d, img_dimensions=%d, 
    init=%s, n_filters=%d, lambda=%e, dropout=%f""" 
    % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
       MP['dim'], init, n_filters, lmbda, drop))
    
    X_true, Y_true = Data['test'][0], Data['test'][1]
 
    Y_preds = model.predict(X_true)
    np.save(MP['dir'] + '/predicted_label.npy', Y_preds)
    score = model.evaluate(X_true, Y_true)
#     print("binary XE score = %f" % (score[0], score[1], score[2]) )
    for j in range(len(score)):
        print("%s: %.2f%%" % (model.metrics_names[j], score[j]*100))
    print(Y_preds)
    print("###################################")
    print("###################################")


def preprocess(Data, dim=128, low=0.1, hi=1.0):
    """Normalize and rescale (and optionally invert) images.
    Parameters
    ----------
    Data : hdf5
        Data array.
    dim : integer, optional
        Dimensions of images, assumes square.
    low : float, optional
        Minimum rescale value. Default is 0.1 since background pixels are 0.
    hi : float, optional
        Maximum rescale value.
    """
    for key in Data:
        print (key)

        Data[key][0] = Data[key][0].reshape(len(Data[key][0]), dim, dim, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.
            # img[img > 0.] = 1. - img[img > 0.]      #inv color
            minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
            img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = img 


if __name__ == '__main__':
    

    # Model Parameters
    MP = {}
    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = '/home/yifanc3/dataset/hackathon_dataset'
    # Image width/height, assuming square images.
    MP['dim'] = 128
    # Batch size: smaller values = less memory but less accurate gradient estimate
    MP['bs'] = 16
    # Number of training epochs.
    MP['epochs'] = 30
    # Save model (binary flag) and directory.
    MP['save_models'] = 1
    MP['save_dir'] = './results/shiru'
    if not os.path.exists(MP['save_dir']):
        os.makedirs(MP['save_dir'])

    
    # Model Parameters (to potentially iterate over, keep in lists).
    MP['N_runs'] = 1                # Number of runs
    MP['filter_length'] = 3       # Filter length
    MP['lr'] = 0.0003             # Learning rate
    MP['n_filters'] = 112        # Number of filters
    MP['init'] = 'he_normal'     # Weight initialization
    MP['lambda'] = 1e-6           # Weight regularization
    MP['dropout'] = 0.15          # Dropout fraction
    
    folder_input = MP['dir'] + '/all_frames_DEM'
    folder_label = MP['dir'] + '/all_masks_5m6b'
    input_data, label_data = select_feature_images(folder_input,folder_label)	
    
    input_train, input_test, label_train, label_test = train_test_split(
            input_data, label_data, test_size=0.1, random_state=21, shuffle=False)

    input_train, input_dev, label_train, label_dev = train_test_split(
            input_train, label_train, test_size=0.1, random_state=21, shuffle=False)	

    MP['n_train'] = int(4064/2)
    MP['n_dev'] = 48
    MP['n_test'] = 240
    
    n_train = MP['n_train'] 
    n_dev = MP['n_dev']
    n_test = MP['n_test'] 
    Data = {
        'train':[input_train[:n_train].astype('float32'),
                 label_train[:n_train].astype('float32')],
        'dev':[input_dev[:n_dev].astype('float32'),
                label_dev[:n_dev].astype('float32')],
        'test':[input_test[:n_test].astype('float32'),
                label_test[:n_test].astype('float32')]
        }

    # Rescale, normalize, add extra dim
    preprocess(Data)
    
    np.save(MP['dir'] + '/real_label.npy', Data['test'][1])
    
    train_and_test_model(Data, MP)
    real = np.load(MP['dir'] + '/real_label.npy')
    pred = np.load(MP['dir'] + '/predicted_label.npy')
    
    predicted_data = np.zeros(pred.shape)
 
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                if (pred[i,j,k]>=0.3):
                    predicted_data[i,j,k] =1
                else:
                    predicted_data[i,j,k] =0
	
    for i in range(100):
        visualize(real,pred,predicted_data,i)

