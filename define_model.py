import os
import numpy as np
from build_model import build_model
from keras.callbacks import ModelCheckpoint,TensorBoard,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from dataGenerator import custom_image_generator
    

def get_callbacks(weights_path, model_path, patience_lr):

    logdir = os.path.join(model_path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    if weights_path:
        mcp_save = ModelCheckpoint(weights_path, save_best_only=False)
        return [mcp_save, tensorboard]
    return [tensorboard]


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

    weights_path = None 
    if opt.save_model:
        weights_path = opt.model_path +'/weights.{epoch:02d}-{val_loss:.2f}-{val_iou_label}.hdf5'
    
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
