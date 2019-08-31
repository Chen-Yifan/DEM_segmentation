import os
import numpy as np
from keras.callbacks import ModelCheckpoint,CSVLogger, EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD

import tensorflow as tf
import keras.backend as K
from models import models
from metrics import *
from losses import *
from k_fold import *
from options.train_options import TrainOptions

#Options
opt = TrainOptions().parse()

BATCH_SIZE = opt.batch_size
NO_OF_EPOCHS = opt.epochs
shape = opt.input_shape
inc = opt.input_channel
aug = opt.augmentation # to decide if shuffle
Model_path = opt.Model_path
Result_path = opt.Result_path
model = opt.model
k = opt.k
frame_path = opt.frame_path
mask_path = opt.mask_path



mkdir(Model_path)
Checkpoint_path = Model_path + 'ckpt_weights/'
mkdir(Checkpoint_path)


# k-fold cross-validation
img, mask = load_data(frame_path, mask_path, shape, inc)
train_list, test_list = k_fold(len(img), k = k)
print(len(train_list), len(test_list))

model_history = [] 

for i in range(k):
    print('====The %s Fold===='%i)
    #shuffle the index
#     random.shuffle(train_list[i])
#     random.shuffle(test_list[i])
    
    train_x = img[train_list[i]]
    train_y = mask[train_list[i]]
    test_x = img[test_list[i]]
    test_y = mask[test_list[i]]
    
    #model 
    input_shape = (shape,shape,inc)
    if(model == 'unet'):
        m = model.unet(input_shape)
    else:
        m = model.segnet(input_shape)
        

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    
    weights = np.array([1.0,200.0])
    loss = weighted_categorical_crossentropy(weights)

    Mean_IOU = Mean_IoU_cl(cl=2)
    
    m.compile( optimizer = opt2, loss = loss, metrics = [per_pixel_acc, Mean_IOU, Mean_IOU_label, precision, recall, f1score])

    #callback
    ckpt_path = Checkpoint_path + '%s/'%i
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    weights_path = ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_Mean_IOU:.2f}.hdf5'
    
    callbacks = get_callbacks(i, weights_path, Model_path, 5)
    
    if(aug):
    # data augmentation
        train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_aug(train_x, train_y, 32, ratio = 0.18)
        history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data=val_gen,
                              validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                              shuffle = True,
                              callbacks=callbacks)
    else:
#         train_gen, val_gen, NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES = train_gen_noaug(train_x, train_y, 32, ratio = 0.18)
        history = m.fit(train_x, train_y, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                         verbose=1, validation_split=0.18, shuffle = True)
    
    model_history.append(history)
    
    # serialize model to JSON
    model_json = m.to_json()
    with open(os.path.join(Model_path,"model%s.json" %i), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("Saved model to disk")
    m.save(os.path.join(Model_path,'model%s.h5' %i))
    
    #TEST
    print('======Start Testing======')

    score = m.evaluate(test_x, test_y, verbose=0)
    print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
    print("%s: %.2f%%" % (m.metrics_names[2], score[2]*100))
    print("%s: %.2f%%" % (m.metrics_names[3], score[3]*100))
    print("%s: %.2f%%" % (m.metrics_names[4], score[4]*100))
    print("%s: %.2f%%" % (m.metrics_names[5], score[5]*100))
    print("%s: %.2f%%" % (m.metrics_names[6], score[6]*100))

    results = m.predict(test_x)
    new_r = np.argmax(results,axis=-1)

    #save image
    result_path = "/home/yifanc3/results/%s/%s/%s"%(date,Model_name,i)

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    print('result:', result_path)
    
    save_result(train_frame_path, result_path, test_list[i], results, test_x, test_y, shape)
    # saveFrame_256(save_frame_path, test_frame_path, X)
    print("======="*12, end="\n\n\n")

