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

from spatial_generator import SDG
#Options
opt = TrainOptions().parse()

BATCH_SIZE = opt.batch_size
NO_OF_EPOCHS = opt.epochs
shape = opt.input_shape
inc = opt.input_channel
aug = opt.augmentation # to decide if shuffle
Model_path = opt.Model_path
result_path = opt.Result_path
model = opt.model
k = opt.k
frame_path = opt.frame_path
mask_path = opt.mask_path
date = opt.date
Model_name = opt.name

mkdir(Model_path)
Checkpoint_path = Model_path + 'ckpt_weights/'
mkdir(Checkpoint_path)


frame_path = '/home/yifanc3/dataset/data/origin/mclean_roi_mb.tif'
mask_path = '/home/yifanc3/dataset/data/origin/cl1.tif'

# load data by spatialDataGenerator
frame_gen, mask_gen, num_files = SDG(frame_path, mask_path, size=(128,128), overlap=0.5, batch_size=32)
print('**********GENERATOR SUCCESS***********')

model_history = [] 
i = 0

#model 
input_shape = (shape,shape,inc)
if(model == 'unet'):
    m = models.unet(input_shape=input_shape)
else:
    m = models.segnet(input_shape=input_shape)
    

'''
    metrics, optimizers and loss 
'''
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)

weights = np.array([1.0,200.0])
loss = weighted_categorical_crossentropy(weights)

Mean_IOU = Mean_IoU_cl(cl=2)

'''compile'''
m.compile( optimizer = opt2, loss = loss, metrics = [per_pixel_acc, Mean_IOU, Mean_IOU_label, precision, recall, f1score])

#callback
ckpt_path = Checkpoint_path
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
weights_path = ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_Mean_IOU:.2f}.hdf5'

callbacks = get_callbacks(i, weights_path, Model_path, 5)
    
'''Fit Generator'''
history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          shuffle = True,
                          callbacks=callbacks)


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
mkdir(result_path)
print('result:', result_path)

save_result(frame_path, result_path, test_list[i], results, test_x, test_y, shape)
# saveFrame_256(save_frame_path, test_frame_path, X)
print("======="*12, end="\n\n\n")

