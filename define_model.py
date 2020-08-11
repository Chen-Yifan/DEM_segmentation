import os
import numpy as np
from keras.callbacks import ModelCheckpoint,TensorBoard,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from dataGenerator import custom_image_generator, val_datagenerator, no_aug_generator
from keras.optimizers import Adadelta, Adam, SGD
from metrics import *
from util.util import *
import tensorflow as tf
from models.models import *
from losses import * 
import lovasz_losses_tf as L

def get_callbacks(weights_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    reduce_lr_loss = ReduceLROnPlateau(factor=0.9)
    if weights_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, reduce_lr_loss, tensorboard]
    return [reduce_lr_loss, tensorboard]

def helper_pred(model, X_true, Y_true, opt):
    # multi-band
    # (X_true, Y_true) = val_datagenerator(X_true, Y_true, opt.use_gradient)
    score = model.evaluate(X_true, Y_true)  
    Y_pred = model.predict(X_true)
    print('shape for skelentonize',Y_pred.shape, Y_true.shape)
    print('***********TEST RESULTS, write to output.txt*************')
    print('result_path',opt.model_path)
    
    message = ''
    for j in range(len(model.metrics_names)):
        message += "%s: %.2f%% \n" % (model.metrics_names[j], score[j]*100)
    # centerline accuracy
    message += "centerlineAccuracy: %.2f%% \n" %(centerline_acc(Y_true, Y_pred)*100)
    print(message)
    
    with open(opt.model_path+'/output_%s.txt'%opt.n_epoch, 'wt') as opt_file:
        opt_file.write(message)
            
    print('********************SAVE RESULTS ************************')
    result_dir = opt.result_path + '/epoch%s/'%opt.n_epoch
    mkdir(result_dir)
    np.save(result_dir + 'pred_labels.npy', Y_pred)  
    
    return Y_pred

def define_model(Data, opt):
    dim = opt.dim
    learn_rate = float(opt.lr)
#     lmbda = opt.lambda
    drop = opt.dropout
    FL = opt.filter_length
    num_filters = opt.num_filters
    n_epoch = opt.n_epoch
    bs = opt.batch_size
    init = opt.weight_init
    input_channel = opt.input_channel
    use_gradient = opt.use_gradient
    
    # different loss function
    if opt.model == 'unet':
        if opt.loss == 'L':
            model = unet(input_channel, learn_rate, num_filters, None)
        else:
            model = unet(input_channel, learn_rate, num_filters)

    elif opt.model == 'unet_rgl':
        if opt.loss == 'L':
            model = unet_rgl(input_channel, learn_rate, num_filters, None)
        else:
            model = unet_rgl(input_channel, learn_rate, num_filters)
        
    else:
        if opt.loss == 'L':
            model = unet_shirui(input_channel, 1e-6, drop, init, num_filters, None, learn_rate)  # L
        else:
            model = unet_shirui(input_channel, 1e-6, drop, init, num_filters, 'sigmoid',learn_rate)
        
#     elif opt.loss == 'cce':
#         model = unet(1,(dim,dim,input_channel),'relu','softmax') 
#     else:
#         model = unet(1,(dim,dim,input_channel),'elu',None) 
        
#     model = DeeplabV2(n_classes=1, input_shape=(dim,dim,input_channel))
#     model = segnet(1,(dim,dim,input_channel),'sigmoid') 
#     model = unet(1,(dim,dim,input_channel),'relu','sigmoid') #lovasz 'elu' None threshold 0 
#     model = unet_shirui(1, (dim,dim,input_channel), 1e-6, drop, init, num_filters, output_mode='sigmoid')
    
    
    weights_path = None 
    if opt.save_model:
        weights_path = opt.model_path +'/weights.{epoch:02d}-{val_loss:.2f}-{val_iou:.2f}.hdf5'
        
        model_json = model.to_json()
        with open(opt.model_path+"/model.json", "w") as json_file:
            json_file.write(model_json)
            
    callbacks = get_callbacks(weights_path, opt.model_path, 5)
    
    n_train, n_test, n_val = len(Data['train'][0]), len(Data['test'][0]), len(Data['val'][0])
    np.save(opt.result_path + '/inputs.npy', Data['test'][0])
    np.save(opt.result_path + '/gt_labels.npy', Data['test'][1])
    
    model.fit_generator(
            # no_aug_generator(Data['train'][0], Data['train'][1],batch_size=bs),
            custom_image_generator(Data['train'][0], Data['train'][1], bs),
            steps_per_epoch= n_train//bs, epochs=n_epoch, verbose=1,
            #validation_data=(Data['val'][0], Data['val'][1]),
            validation_data=val_datagenerator(Data['val'][0], Data['val'][1], use_gradient),  # no gen
            validation_steps= n_val,
            callbacks=callbacks)
    
    print('***********FINISH TRAIN & START TESTING******************')
    X_true, Y_true = Data['test'][0], Data['test'][1]
    
    Y_pred = helper_pred(model, X_true, Y_true, opt)
    return X_true, Y_true, Y_pred

def find_weight_dir(opt):
    #file_name[8:10] = the epoch
    weights = os.listdir(opt.model_path)
    print('model_path',opt.model_path)
    for i in weights:
        print(i)
        if 'weights'in i and int(i[8:10]) == opt.n_epoch:
            return os.path.join(opt.model_path,i)

def test_model(opt):
    
    json_path = opt.model_path + '/model.json'
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    #load model and weights
    weight_dir = find_weight_dir(opt)
    print('load weight from:', weight_dir)
    model = model_from_json(loaded_model_json, custom_objects = 
                    {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D})
    model.load_weights(weight_dir)
    
    learn_rate = opt.lr
    optimizer = Adam(lr=learn_rate)
    
    if opt.loss=='bce':
        model.compile(loss='binary_crossentropy', metrics=[iou_label(),per_pixel_acc(),accuracy()], optimizer=optimizer)
    elif opt.loss=='cce':
        model.compile(loss=sparse_softmax_cce, metrics=[iou_label(),per_pixel_acc(),accuracy()], optimizer=optimizer)
    else:
        model.compile(loss=L.lovasz_loss, metrics=[iou_label(threshold=0),per_pixel_acc(threshold=0),accuracy(threshold=0)], optimizer=optimizer)
        
    model.summary()
    
    print('***********FINISH TRAIN & START TESTING******************')
    X_true = np.load(opt.result_path + '/inputs.npy')
    Y_true = np.load(opt.result_path + '/gt_labels.npy') 
    print(X_true.shape, Y_true.shape)
    
    Y_pred = helper_pred(model, X_true, Y_true, opt)
    
    return X_true, Y_true, Y_pred
    

