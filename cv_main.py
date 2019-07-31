from generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os
import tensorflow as tf
import keras.backend as K
import model
from utils import *
from metrics import *
from losses import *
import numpy as np
from keras.models import Model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
import time
from functools import *
from k_fold import *
from keras.models import model_from_json


#hyperparameters
date = '7.11'
BATCH_SIZE = 32
NO_OF_EPOCHS = 120
shape = 128
aug = True # to decide if shuffle
Model_name = '128overlap_300w_unetAdal_120ep_10m6b_noshuffle_renorm_aug'
network = 'unet'
k = 2
band = 6
norm = True

print('batch_size:', BATCH_SIZE, '\ndate:', date, '\nshape:', shape, '\naug:',aug, '\nModel_name', Model_name, '\nk:',k, '; band:', band, '\nnorm:', norm)
    
#Train the model with K-fold Cross Val
#TRAIN
train_frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b/'
train_mask_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_10m6b/'


Model_path = '/home/yifanc3/models/%s/%s/' % (date,Model_name)
if not os.path.isdir(Model_path):
    os.makedirs(Model_path)
    
Checkpoint_path = Model_path + 'ckpt_weights/'
if not os.path.isdir(Checkpoint_path):
    os.makedirs(Checkpoint_path)


# k-fold cross-validation
img, mask = load_data(train_frame_path, train_mask_path, shape, band, norm)
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
    if(network == 'unet'):
        m = model.get_unet(input_shape = (shape,shape,band))
    else:
        m = model.segnet(input_shape = (shape,shape,band))
        

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.1)
    m.compile( optimizer = opt2, loss = pixel_wise_loss, metrics = [per_pixel_acc, Mean_IOU, precision, recall, f1score])

    #callback
    ckpt_path = Checkpoint_path + '%s/'%i
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    weights_path = ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_Mean_IOU:.2f}.hdf5'
    
    callbacks = get_callbacks(weights_path, Model_path, 5)
    
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
    # print("%s: %.2f%%" % (m.metrics_names[6], score[6]*100))

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

    
plt.title('Train MeanIoU vs Val MeanIoU')
plt.plot(model_history[0].history['Mean_IOU'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_Mean_IOU'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['Mean_IOU'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_Mean_IOU'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['Mean_IOU'], label='Train Accuracy Fold 3', color='green', )
# plt.plot(model_history[2].history['val_Mean_IOU'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['Mean_IOU'], label='Train Accuracy Fold 4', color='c', )
# plt.plot(model_history[3].history['val_Mean_IOU'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['Mean_IOU'], label='Train Accuracy Fold 5', color='y', )
# plt.plot(model_history[4].history['val_Mean_IOU'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValMeanIOU.png'))
plt.clf()
plt.cla()
plt.close()


plt.title('Train Acc vs Val Acc')
plt.plot(model_history[0].history['per_pixel_acc'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_per_pixel_acc'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['per_pixel_acc'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_per_pixel_acc'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['per_pixel_acc'], label='Train Accuracy Fold 3', color='green', )
# plt.plot(model_history[2].history['val_per_pixel_acc'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['per_pixel_acc'], label='Train Accuracy Fold 4', color='c', )
# plt.plot(model_history[3].history['val_per_pixel_acc'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['per_pixel_acc'], label='Train Accuracy Fold 5', color='y', )
# plt.plot(model_history[4].history['val_per_pixel_acc'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValper_pixel_acc.png'))
plt.clf()
plt.cla()
plt.close()

plt.title('Train Precision vs Val Precision')
plt.plot(model_history[0].history['precision'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_precision'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['precision'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_precision'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['precision'], label='Train Accuracy Fold 3', color='green', )
# plt.plot(model_history[2].history['val_precision'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['precision'], label='Train Accuracy Fold 4', color='c', )
# plt.plot(model_history[3].history['val_precision'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['precision'], label='Train Accuracy Fold 5', color='y', )
# plt.plot(model_history[4].history['val_precision'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValPrecision.png'))
plt.clf()
plt.cla()
plt.close()

plt.title('Train Precision vs Val recall')
plt.plot(model_history[0].history['recall'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_recall'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['recall'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_recall'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['recall'], label='Train Accuracy Fold 3', color='green', )
# plt.plot(model_history[2].history['val_recall'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['recall'], label='Train Accuracy Fold 4', color='c', )
# plt.plot(model_history[3].history['val_recall'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['recall'], label='Train Accuracy Fold 5', color='y', )
# plt.plot(model_history[4].history['val_recall'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValRecall.png'))
plt.clf()
plt.cla()
plt.close()

plt.title('Train F1 vs Val F1')
plt.plot(model_history[0].history['f1score'], label='Train f1score Fold 1', color='black')
plt.plot(model_history[0].history['val_f1score'], label='Val f1score Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['f1score'], label='Train f1score Fold 2', color='red', )
plt.plot(model_history[1].history['val_f1score'], label='Val f1score Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['f1score'], label='Train f1score Fold 3', color='green', )
# plt.plot(model_history[2].history['val_f1score'], label='Val f1score Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['f1score'], label='Train f1score Fold 4', color='c', )
# plt.plot(model_history[3].history['val_f1score'], label='Val f1score Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['f1score'], label='Train f1score Fold 5', color='y', )
# plt.plot(model_history[4].history['val_f1score'], label='Val f1score Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValF1score.png'))
plt.clf()
plt.cla()
plt.close()
plt.title('Train Loss vs Val Loss')
plt.plot(model_history[0].history['loss'], label='Train loss Fold 1', color='black')
plt.plot(model_history[0].history['val_loss'], label='Val loss Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['loss'], label='Train loss Fold 2', color='red', )
plt.plot(model_history[1].history['val_loss'], label='Val loss Fold 2', color='red', linestyle = "dashdot")
# plt.plot(model_history[2].history['loss'], label='Train loss Fold 3', color='green', )
# plt.plot(model_history[2].history['val_loss'], label='Val loss Fold 3', color='green', linestyle = "dashdot")
# plt.plot(model_history[3].history['loss'], label='Train loss Fold 4', color='c', )
# plt.plot(model_history[3].history['val_loss'], label='Val loss Fold 4', color='c', linestyle = "dashdot")
# plt.plot(model_history[4].history['loss'], label='Train loss Fold 5', color='y', )
# plt.plot(model_history[4].history['val_loss'], label='Val loss Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValLoss.png'))
plt.clf()
plt.cla()
plt.close()