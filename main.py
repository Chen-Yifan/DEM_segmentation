from custom_generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import model
from utils import *
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers.core import Dropout, Activation
from keras.models import Model
from keras.optimizers import Adadelta, Adam


BATCH_SIZE = 32
NO_OF_EPOCHS = 2

#TRAIN
train_frame_path = '/home/yifanc3/dataset/npy/train_frames'
train_mask_path = '/home/yifanc3/dataset/npy/train_masks'

val_frame_path = '/home/yifanc3/dataset/npy/val_frames'
val_mask_path = '/home/yifanc3/dataset/npy/val_masks'

train_gen = data_gen(train_frame_path,train_mask_path, batch_size = BATCH_SIZE)
val_gen = data_gen(val_frame_path,val_mask_path, batch_size = BATCH_SIZE)

# Train the model
NO_OF_TRAINING_IMAGES = len(os.listdir('/home/yifanc3/dataset/npy/train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir('/home/yifanc3/dataset/npy/val_frames/'))

#NO_OF_EPOCHS = 'ANYTHING FROM 30-100 FOR SMALL-MEDIUM SIZED DATASETS IS OKAY'

#BATCH_SIZE = 'BATCH SIZE PREVIOUSLY INITIALISED'

weights_path = '/home/yifanc3/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

#vgg16_32s
inputs = Input((256,256,5))
base = model.get_fcn_vgg16_32s(inputs)
# softmax
#reshape= Reshape((-1,2))(base)
act = Activation('softmax')(base)
m = Model(inputs=inputs, outputs=act)
m.summary()

# opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# m.compile(loss=soft_dice_loss,
#                optimizer=opt,
#                metrics=[Mean_IOU])
m.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = [Mean_IOU])

checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
                             min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          callbacks=callbacks_list)
m.save('Model.h5')

#TEST
print('======Start Testing======')
   
test_frame_path = '/home/yifanc3/dataset/npy/test_frames'
test_mask_path = '/home/yifanc3/dataset/npy/test_masks'

saveMask_256("../results/orig_mask", test_mask_path)
# test_gene = test_gen(test_frame_path, test_mask_path)
# results = m.predict_generator(test_gene, 30, verbose=1)

X,Y = test_gen(test_frame_path, test_mask_path)

score = m.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
# print("%s: %.2f%%" % (m.metrics_names[2], score[2]*100))
# print("%s: %.2f%%" % (m.metrics_names[3], score[3]*100))
# print("%s: %.2f%%" % (m.metrics_names[4], score[4]*100))
# print("%s: %.2f%%" % (m.metrics_names[5], score[5]*100))
# print("%s: %.2f%%" % (m.metrics_names[6], score[6]*100))

results = m.predict(X)
print(np.unique(results))
#save image
saveResult("../results/v1",test_mask_path,results)











