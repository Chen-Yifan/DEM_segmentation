import os
import numpy as np
from keras.callbacks import ModelCheckpoint,TensorBoard,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from dataGenerator import custom_image_generator, val_datagenerator, no_aug_generator
from keras.optimizers import Adadelta, Adam, SGD
from metrics import *
from losses import * 
from util.util import *
import tensorflow as tf
from models.models import *
import lovasz_losses_tf as L
import segmentation_models as sm
# from tensorflow.keras.metrics import MeanIoU, Precision, Recall, BinaryAccuracy

########################### Helper functions ###############################################

def get_callbacks(weights_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=5, verbose=1)
    if weights_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, reduce_lr_loss, tensorboard]
    return [reduce_lr_loss, tensorboard]

def helper_pred(model, X_true, Y_true, opt):
    # model.compile(loss='binary_crossentropy', metrics=[iou_label(),per_pixel_acc(),accuracy(),recall_m, precision_m, f1_m], optimizer=Adam(1e-4))
    # multi-band
    (X_true, Y_true) = val_datagenerator(X_true, Y_true, opt.use_gradient, opt.DEM_only) # no gen
    # if '3b' in opt.ckpt_name or opt.use_gradient==1:
    #     print('use_gradient==True')
    #     (X_true, Y_true) = val_datagenerator(X_true, Y_true, opt.use_gradient)
    score = model.evaluate(X_true, Y_true)  
    Y_pred = model.predict(X_true)
    print('shape for skelentonize',Y_pred.shape, Y_true.shape)
    print('***********TEST RESULTS, write to output.txt*************')
    print('result_path',opt.model_path)
    print('epoch: %d'%opt.n_epoch)
    
    message = ''
    for j in range(len(model.metrics_names)):
        message += "%s: %.2f%% \n" % (model.metrics_names[j], score[j]*100)
    # centerline accuracy
    # message += "centerlineAccuracy: %.2f%% \n" %(centerline_acc(Y_true, Y_pred)*100)
    print(message)
    
    with open(opt.model_path+'/output_%s.txt'%opt.n_epoch, 'wt') as opt_file:
        opt_file.write(message)
            
    print('********************SAVE RESULTS ************************')
    result_dir = opt.result_path + '/epoch%s/'%opt.n_epoch
    mkdir(result_dir)
    np.save(result_dir + 'pred_labels.npy', Y_pred)
    
    return Y_pred

def choose_model(opt):
    dim = opt.dim
    learn_rate = opt.lr
    lmbda = opt.lmbda
    drop = opt.dropout
    FL = opt.filter_length
    num_filters = opt.num_filters
    init = opt.weight_init
    input_channel = opt.input_channel
    if opt.pretrained_weights!= '':
        pretrained_weights = opt.pretrained_weights
    else: 
        pretrained_weights = None
    
    # different model and loss function
    if opt.model == 'unet':
        if opt.loss=='L':
            model = unet(input_channel, learn_rate, num_filters,'elu', opt.loss, None, pretrained_weights)
            # model = lovasz_unet(1,(dim,dim,input_channel),'elu',None) 
        else:
            model = unet(input_channel, learn_rate, num_filters,'relu', opt.loss, 'sigmoid', pretrained_weights)
        # if opt.loss == 'L':
        #     model = unet(input_channel, learn_rate, num_filters, None)
        # else:
        #     model = unet(input_channel, learn_rate, num_filters)

    elif opt.model == 'unet_rgl':
        if opt.loss == 'L':
            print("=====unet_rgl Lovasz")
            model = unet_rgl(input_channel, learn_rate, num_filters, pretrained_weights, None)
        else:
            model = unet_rgl(input_channel, learn_rate, num_filters, pretrained_weights)
        
    elif opt.model == 'resnet':
        model = sm.Unet('resnet34', input_shape=(128, 128, 1), encoder_weights=None, classes=1, activation='sigmoid')
        model.compile(loss='binary_crossentropy', metrics=[
                      iou_label(), per_pixel_acc(), accuracy()], optimizer=Adam(learn_rate))
    else:
        if opt.loss == 'L':
            model = unet_shirui(input_channel, lmbda, drop, init, num_filters, opt.loss, None, learn_rate)  # L
        else:
            model = unet_shirui(input_channel, lmbda, drop, init, num_filters, opt.loss, 'sigmoid',learn_rate)

    return model

def find_weight_dir(opt):
    weights = os.listdir(opt.model_path)
    for name in weights:
        if 'weights' in name:
            epoch = name.split('.')[1].split('-')[0]
            if int(epoch) == opt.n_epoch:
                print(epoch,name)
                return os.path.join(opt.model_path,name)
                continue
            
##############################################################################################################

def train_model(Data, opt):
    dim = opt.dim
    n_epoch = opt.n_epoch
    bs = opt.batch_size
    use_gradient = opt.use_gradient
    DEM_only = opt.DEM_only
    
    model = choose_model(opt)   # choose model based on options
    
    """ save model: model.json """
    weights_path = None 
    if opt.save_model:
        weights_path = opt.model_path +'/weights.{epoch:02d}-{val_loss:.4f}-{val_iou:.4f}.hdf5'
        
        model_json = model.to_json()
        with open(opt.model_path+"/model.json", "w") as json_file:
            json_file.write(model_json)
            
    callbacks = get_callbacks(weights_path, opt.model_path, 5)
    
    """Fit data/generator to model"""
    n_train, n_test, n_val = len(Data['train'][0]), len(Data['test'][0]), len(Data['val'][0])
    
    model.fit_generator(
            # no_aug_generator(Data['train'][0], Data['train'][1],bs, use_gradient),
            custom_image_generator(Data['train'][0], Data['train'][1], bs, use_gradient, DEM_only),
            steps_per_epoch= n_train//bs, epochs=n_epoch, verbose=1,
            # validation_data=(Data['val'][0], Data['val'][1]),
            validation_data=val_datagenerator(Data['val'][0], Data['val'][1], use_gradient, DEM_only),  # no gen
            validation_steps= n_val,
            callbacks=callbacks)
            
    
    print('***********FINISH TRAIN & START TESTING******************')
    X_true, Y_true = Data['test'][0], Data['test'][1]
    
    Y_pred = helper_pred(model, X_true, Y_true, opt)
    return X_true, Y_true, Y_pred


def test_model(opt):
    
    json_path = opt.model_path + '/model.json'
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    #load model and weights
    weight_dir = find_weight_dir(opt)
    # weight_dir = os.path.join(opt.model_path,'weights.111-0.0828-0.2607.hdf5')
    print('load weight from:', weight_dir)
    model = model_from_json(loaded_model_json)  # for segnet: custom_objects = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D})
    model.load_weights(weight_dir)
    
    learn_rate = opt.lr
    optimizer = Adam(lr=learn_rate)
    
    if opt.loss=='bce':
        model.compile(loss='binary_crossentropy', metrics=[iou_label(),dice_coefficient(),accuracy(),recall_m, precision_m, f1_m], optimizer=optimizer)
    elif opt.loss=='cce':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(),dice_coefficient(),accuracy(),recall_m, precision_m, f1_m], optimizer=optimizer)
    elif opt.loss=='T':
        model.compile(loss=FocalTverskyLoss, metrics=[iou_label(),dice_coefficient(),accuracy(),recall_m, precision_m, f1_m], optimizer=optimizer)
    else:
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(),dice_coefficient(),accuracy(),recall_m, precision_m, f1_m], optimizer=optimizer)
        
    model.summary()
    
    print('***********FINISH TRAIN & START TESTING******************')
    X_true = np.load(opt.result_path + '/inputs.npy')
    Y_true = np.load(opt.result_path + '/gt_labels_ext.npy') 
    print(X_true.shape, Y_true.shape)
    
    Y_pred = helper_pred(model, X_true, Y_true, opt)
    
    return X_true, Y_true, Y_pred
    

