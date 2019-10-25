import os
import numpy as np
from build_model import build_model
from keras.callbacks import ModelCheckpoint,TensorBoard,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from dataGenerator import custom_image_generator
from keras.optimizers import Adadelta, Adam, SGD
from metrics import iou_label,per_pixel_acc
    

def get_callbacks(weights_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    if weights_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, tensorboard]
    return [tensorboard]


def define_model(Data, opt):
    dim = int(opt.input_shape),
    learn_rate = opt.lr
#     lmbda = opt.lambda
    drop = opt.dropout
    FL = opt.filter_length
    num_filters = opt.num_filters
    n_epoch = opt.n_epoch
    bs = opt.batch_size
    init = opt.weight_init
    
    model = build_model(dim, learn_rate, 1e-6, drop, FL, init, num_filters)

    weights_path = None 
    if opt.save_model:
        weights_path = opt.model_path +'/weights.{epoch:02d}-{val_loss:.2f}-{val_iou_label:.2f}.hdf5'
    
    callbacks = get_callbacks(weights_path, opt.model_path, 5)
    
    n_train, n_test, n_val = len(Data['train'][0]), len(Data['test'][0]), len(Data['val'][0])
    
    model.fit_generator(
            #(Data['train'][0], Data['train'][1]),
            custom_image_generator(Data['train'][0], Data['train'][1],
                                   batch_size=bs),
            steps_per_epoch= n_train//bs, epochs=n_epoch, verbose=1,
            validation_data=(Data['val'][0],Data['val'][1]), #no gen
            validation_steps= n_val,
            callbacks=callbacks)
    
    if opt.save_model:
        model_json = model.to_json()
    with open(opt.model_path+"/model.json", "w") as json_file:
        json_file.write(model_json)

    print('***********FINISH TRAIN & START TESTING******************')
    X_true, Y_true = Data['test'][0], Data['test'][1]
    
    score = model.evaluate(X_true, Y_true)    
    print('***********TEST RESULTS, write to output.txt*************')
    message = ''
    for j in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (model.metrics_names[j], score[j]*100))
        message += "%s: %.2f%% \n" % (model.metrics_names[j], score[j]*100)
        
    with open(opt.model_path+'/output.txt', 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
            
    print('********************SAVE RESULTS ************************')
    Y_preds = model.predict(X_true)
    np.save(opt.result_path + '/gt_labels.npy', Data['test'][1])
    np.save(opt.result_path + '/pred_labels.npy', Y_preds)        
    
    print('==================FINISH WITHOUT ERROR===================')
#     return Y_preds

def find_weight_dir(opt):
    #file_name[8:10] = the epoch
    weights = os.listdir(opt.model_path)
    print('model_path',opt.model_path)
    for i in weights:
        print(i)
        if 'weights'in i and int(i[8:10]) == opt.n_epoch:
            return os.path.join(opt.model_path,i)

def test_model(Data, opt):
    
    json_path = opt.model_path + '/model.json'
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    #load model and weights
    weight_dir = find_weight_dir(opt)
    print(weight_dir)
    print(weight_dir)
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_dir)
    
    learn_rate = opt.lr
    optimizer = Adam(lr=learn_rate)
    
    model.compile(loss='binary_crossentropy', metrics=[iou_label,per_pixel_acc,'accuracy'], optimizer=optimizer)
    model.summary()
    
    print('***********FINISH TRAIN & START TESTING******************')
    X_true, Y_true = Data['test'][0], Data['test'][1]
    
    score = model.evaluate(X_true, Y_true)    
    print('***********TEST RESULTS, write to output.txt*************')
    message = ''
    for j in range(len(model.metrics_names)):
        print("%s: %.2f%%" % (model.metrics_names[j], score[j]*100))
        message += "%s: %.2f%% \n" % (model.metrics_names[j], score[j]*100)
        
    with open(opt.model_path+'/output.txt', 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
            
    print('********************SAVE RESULTS ************************')
    Y_preds = model.predict(X_true)
    np.save(opt.result_path + '/gt_labels.npy', Data['test'][1])
    np.save(opt.result_path + '/pred_labels.npy', Y_preds)        
    
    print('==================FINISH WITHOUT ERROR===================')
#     return Y_preds
    

