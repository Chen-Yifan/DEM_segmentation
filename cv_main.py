from generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os
import tensorflow as tf
import keras.backend as K
import model
from utils import *
import numpy as np
from keras.models import Model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
import time
from functools import *
from k_fold import *


#hyperparameters
BATCH_SIZE = 32
NO_OF_EPOCHS = 100
Model_name = '800w_segnetAdal_cv5_128_100ep'
result_name = '800w_segnetAdal_cv5_128_100ep'
k = 1
    
#Train the model with K-fold Cross Val
    
#TRAIN
train_frame_path = '/home/yifanc3/dataset/data/selected_128/all_frames'
train_mask_path = '/home/yifanc3/dataset/data/selected_128/all_masks'

Checkpoint_path = '/home/yifanc3/checkpoints/%s/' % Model_name
if not os.path.isdir(Checkpoint_path):
    os.makedirs(Checkpoint_path)    
Model_path = '/home/yifanc3/models/%s/' % Model_name
if not os.path.isdir(Model_path):
    os.makedirs(Model_path)
    
weights_path = '/home/yifanc3/checkpoints/%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5' % Model_name

# k-fold cross-validation
img, mask = load_data(train_frame_path, train_mask_path)
train_list, test_list = k_fold(len(img), k = k)
print(len(train_list), len(test_list))

model_history = [] 

for i in range(k):
    print('====The %s Fold===='%i)
    NO_OF_TRAINING_IMAGES = len(train_list[i])
    NO_OF_TEST_IMAGES = len(test_list[i])
    
    train_x = img[train_list[i]]
    train_y = mask[train_list[i]]
    test_x = img[test_list[i]]
    test_y = mask[test_list[i]]
    
    # data augmentation
    
    #model 
    m = model.segnet(input_shape = (128,128,5))

    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    m.compile( optimizer = opt2, loss = pixel_wise_loss, metrics = [per_pixel_acc, Mean_IOU, precision, recall, f1score])

    #callback
    name_weights = "cv_model" + str(i) + "_weights.h5"
    name_weights = os.path.join(Model_path,name_weights)
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=5)
#     history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
#                               steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
#                               validation_data=val_gen,
#                               validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
#                               shuffle = True,
#                               callbacks=callbacks)
    history = m.fit(train_x, train_y, epochs=NO_OF_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                         verbose=1, validation_split=0.1)
    
    model_history.append(history)
    
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
    result_path = "/home/yifanc3/results/%s/%s"%(result_name,i)

    if not os.path.isdir(result_path):
        os.makedirs(result_path)


    save_result(train_frame_path, result_path, test_list[i], results, test_x, test_y)
    # saveFrame_256(save_frame_path, test_frame_path, X)
    print("======="*12, end="\n\n\n")

    
plt.title('Train MeanIoU vs Val MeanIoU')
plt.plot(model_history[0].history['Mean_IOU'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[0].history['val_Mean_IOU'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['Mean_IOU'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[1].history['val_Mean_IOU'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history[2].history['Mean_IOU'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[2].history['val_Mean_IOU'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history[3].history['Mean_IOU'], label='Train Accuracy Fold 4', color='c', )
plt.plot(model_history[3].history['val_Mean_IOU'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
plt.plot(model_history[4].history['Mean_IOU'], label='Train Accuracy Fold 5', color='y', )
plt.plot(model_history[4].history['val_Mean_IOU'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
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
plt.plot(model_history[2].history['per_pixel_acc'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[2].history['val_per_pixel_acc'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history[3].history['per_pixel_acc'], label='Train Accuracy Fold 4', color='c', )
plt.plot(model_history[3].history['val_per_pixel_acc'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
plt.plot(model_history[4].history['per_pixel_acc'], label='Train Accuracy Fold 5', color='y', )
plt.plot(model_history[4].history['val_per_pixel_acc'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
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
plt.plot(model_history[2].history['precision'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[2].history['val_precision'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history[3].history['precision'], label='Train Accuracy Fold 4', color='c', )
plt.plot(model_history[3].history['val_precision'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
plt.plot(model_history[4].history['precision'], label='Train Accuracy Fold 5', color='y', )
plt.plot(model_history[4].history['val_precision'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
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
plt.plot(model_history[2].history['recall'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[2].history['val_recall'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history[3].history['recall'], label='Train Accuracy Fold 4', color='c', )
plt.plot(model_history[3].history['val_recall'], label='Val Accuracy Fold 4', color='c', linestyle = "dashdot")
plt.plot(model_history[4].history['recall'], label='Train Accuracy Fold 5', color='y', )
plt.plot(model_history[4].history['val_recall'], label='Val Accuracy Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValRecall.png'))
plt.clf()
plt.cla()
plt.close()

plt.title('Train Loss vs Val Loss')
plt.plot(model_history[0].history['loss'], label='Train loss Fold 1', color='black')
plt.plot(model_history[0].history['val_loss'], label='Val loss Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['loss'], label='Train loss Fold 2', color='red', )
plt.plot(model_history[1].history['val_loss'], label='Val loss Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history[2].history['loss'], label='Train loss Fold 3', color='green', )
plt.plot(model_history[2].history['val_loss'], label='Val loss Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history[3].history['loss'], label='Train loss Fold 4', color='c', )
plt.plot(model_history[3].history['val_loss'], label='Val loss Fold 4', color='c', linestyle = "dashdot")
plt.plot(model_history[4].history['loss'], label='Train loss Fold 5', color='y', )
plt.plot(model_history[4].history['val_loss'], label='Val loss Fold 5', color='y', linestyle = "dashdot")
plt.legend()
plt.savefig(os.path.join(Model_path,'TrainValLoss.png'))
plt.clf()
plt.cla()
plt.close()




