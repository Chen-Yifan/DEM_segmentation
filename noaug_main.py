from custom_generator import *
from generator import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import model
from model import dice_coef_loss, dice_coef
from utils import *
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from keras.layers.core import Dropout, Activation
from keras.models import Model
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
import time
from functools import *

BATCH_SIZE = 32
NO_OF_EPOCHS = 100
Model_name = '500w_unetAdam_100ep'
result_name = '500w_unetAdam_100ep'

Checkpoint_path = '/home/yifanc3/checkpoints/%s/' % Model_name
if not os.path.isdir(Checkpoint_path):
    os.makedirs(Checkpoint_path)    
    
#TRAIN
train_frame_path = '/home/yifanc3/dataset/npy/no_shuffle/train_frames'
train_mask_path = '/home/yifanc3/dataset/npy/no_shuffle/train_masks'

val_frame_path = '/home/yifanc3/dataset/npy/no_shuffle/val_frames'
val_mask_path = '/home/yifanc3/dataset/npy/no_shuffle/val_masks'

train_gen = data_gen(train_frame_path,train_mask_path, batch_size = BATCH_SIZE)
val_gen = data_gen(val_frame_path,val_mask_path, batch_size = BATCH_SIZE)
# train_gen = train_gen_aug(train_frame_path, train_mask_path, BATCH_SIZE)
# val_gen = val_gen_aug(val_frame_path, val_mask_path, BATCH_SIZE)

# Train the model
NO_OF_TRAINING_IMAGES = len(os.listdir('/home/yifanc3/dataset/npy/no_shuffle/train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir('/home/yifanc3/dataset/npy/no_shuffle/val_frames/'))

#NO_OF_EPOCHS = 'ANYTHING FROM 30-100 FOR SMALL-MEDIUM SIZED DATASETS IS OKAY'

#BATCH_SIZE = 'BATCH SIZE PREVIOUSLY INITIALISED'

weights_path = '/home/yifanc3/checkpoints/%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5' % Model_name

#vgg16_32s
inputs = Input((256,256,5))
base = model.get_unet(inputs)
# print('base shape,',base.get_shape())
m = Model(inputs=inputs, outputs=base)
m.summary()

opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0)

# m.compile(loss=soft_dice_loss,
#                optimizer=opt,
#                metrics=[Mean_IOU])


start = time.time()

ncce = partial(w_categorical_crossentropy)

m.compile( optimizer = opt, loss = pixel_wise_loss, metrics = [Mean_IOU])

checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
                             min_delta = 0.01, patience = 3, mode = 'min')

callbacks_list = [checkpoint, csv_logger] #, earlystopping]

history = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          callbacks=callbacks_list)


Model_path = '/home/yifanc3/models/%s/' % Model_name
if not os.path.isdir(Model_path):
    os.makedirs(Model_path)
    
    
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['Mean_IOU'])
plt.plot(history.history['val_Mean_IOU'])
plt.title('model Mean_IOU')
plt.ylabel('Mean_IOU')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(Model_path,'Mean_IoU.png'))
plt.clf()
plt.cla()
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(Model_path,'loss.png'))


Model_path = '/home/yifanc3/models/%s/' % Model_name
if not os.path.isdir(Model_path):
    os.makedirs(Model_path)

m.save(os.path.join(Model_path,'%s.h5' %Model_name))
end = time.time()
print('train_time = %d' %(end - start))

#TEST
print('======Start Testing======')
   
test_frame_path = '/home/yifanc3/dataset/npy/no_shuffle/test_frames'
test_mask_path = '/home/yifanc3/dataset/npy/no_shuffle/test_masks'


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
new_r = np.argmax(results,axis=-1)

#save image
save_frame_path = "/home/yifanc3/results/orig_frames_256"
result_path = os.path.join("/home/yifanc3/results/",result_name)

if not os.path.isdir(result_path):
    os.makedirs(result_path)
    
if not os.path.isdir(save_frame_path):
    os.makedirs(save_frame_path)
    
saveResult(result_path, test_mask_path,results)
# saveFrame_256(save_frame_path, test_frame_path, X)










