from keras.metrics import binary_accuracy
from keras.models import Model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam, SGD

from metrics import iou_label,per_pixel_acc
from dataGenerator import custom_image_generator

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
    

def get_callbacks(weight_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    if weight_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, tensorboard]
    return [tensorboard]
            
    
    
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
    model.compile(loss='binary_crossentropy', metrics=[iou_label,per_pixel_acc], optimizer=optimizer)
    model.summary()

    return model


def define_model(Data, opt):
    dim = opt.input_shape,
    learn_rate = opt.lr
#     lmbda = opt.lambda
    drop = opt.dropout
    FL = opt.filter_length
    num_filters = opt.num_filters
    n_epoch = opt.n_epoch
    bs = opt.batch_size
    init = opt.weight_init
    
    model = build_model(dim, learn_rate, 1e-6, drop, FL, init, num_filters)
    
    if opt.save_model:
        weights_path = model_path +'/weights.{epoch:02d}-{val_loss:.2f}-{val_iou_label}.hdf5'
    else: weigths_path = None
    
    callbacks = get_callbacks(i, opt.optimizer, weights_path, Model_path, 5)
    
    n_train, n_test, n_val = len(Data['train'][0]), len(Data['test'][0]), len(Data['val'][0])
    
    model.fit_generator(
            custom_image_generator(Data['train'][0], Data['train'][1],
                                   batch_size=bs),
            steps_per_epoch= n_train//bs, epochs=n_epoch, verbose=1,
            validation_data=(Data['val'][0],Data['val'][1]), #no gen
            validation_steps= n_test,
            callbacks=callbacks)
    
    if opt.save_model:
        model_json = model.to_json()
    with open(model_path+"/model.json", "w") as json_file:
        json_file.write(model_json)

    print('***********FINISH TRAIN & START TESTING******************')
    X_true, Y_true = Data['test'][0], Data['test'][1]
    
    score = model.evaluate(X_true, Y_true)    
    print('***********TEST RESULTS, write to output.txt*************')
    message = ''
    for j in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (m.metrics_names[j], score[j]*100))
        message += "%s: %.2f%% \n" % (m.metrics_names[j], score[j]*100)
        
    with open(model_path+'/output.txt', 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
            
    print('********************SAVE RESULTS ************************')
    Y_preds = model.predict(X_true)
    np.save(opt.result_path + '/gt_labels.npy', Data['test'][1])
    np.save(opt.result_path + '/pred_labels.npy', Y_preds)        
    
    print('==================FINISH WITHOUT ERROR===================')
