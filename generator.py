from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
#from scipy.misc import imresize
import os
import random

# def trainvalGenerator(train_frame_path, train_mask_path, val_frame_path, val_mask_path, BATCH_SIZE):
#     train_datagen = ImageDataGenerator(
#             rescale=1./300,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True)

#     val_datagen = ImageDataGenerator(rescale=1./300)

#     train_image_generator = train_datagen.flow_from_directory(
#     directory = train_frame_path,
#     class_mode=None,
#     color_mode= 'rgba',
#     batch_size = BATCH_SIZE)

#     train_mask_generator = train_datagen.flow_from_directory(
#     directory = train_mask_path,
#     class_mode="categorical",
#     color_mode= 'grayscale',
#     batch_size = BATCH_SIZE)

#     val_image_generator = val_datagen.flow_from_directory(
#     directory = val_frame_path,
#     class_mode=None,
#     color_mode= 'rgba',
#     batch_size = BATCH_SIZE)


#     val_mask_generator = val_datagen.flow_from_directory(
#     directory = val_mask_path,
#     class_mode = "categorical",
#     color_mode= 'grayscale',
#     batch_size = BATCH_SIZE)


#     print('val_mask_generator',val_mask_generator)
#     train_generator = zip(train_image_generator, train_mask_generator)
#     val_generator = zip(val_image_generator, val_mask_generator)
    
#     return train_generator, val_generator


def train_gen_aug(img_folder, mask_folder, batch_size):
    
#     datagen = ImageDataGenerator(
#     featurewise_center=True,
# #     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
    data_gen_args = dict(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10
                        )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    
    img = np.zeros((len(n), 256, 256, 5)).astype(np.float32)
    mask = np.zeros((len(n), 256, 256, 2), dtype=np.float32)

    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        #train_img =  cv2.resize(train_img, (256, 256))# Read an image from folder and resize
        train_img = np.zeros((256,256,5))
        #resize
        for a in range(5):
            train_img[:,:,a] = cv2.resize(train_img_0[:,:,a], (256, 256))
        img[i] = train_img #add to array - img[0], img[1], and so on.
        
        #train_mask
        train_mask_0 = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0
        train_mask = np.where(train_mask_0==2.0, 0.0, train_mask_0) 
        #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
        train_mask = cv2.resize(train_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
        mask[i,:,:,0] = np.squeeze(train_mask)
        mask[i,:,:,1] = np.squeeze(1-train_mask)
        
    #print(mask.shape)
#     mask = mask.reshape(len(n),256*256, 2)

    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    seed = 2018
    
#     image_datagen.fit(img, augment=True, seed=seed)
#     mask_datagen.fit(mask, augment=True, seed=seed) # mask and image separate
    
    img_gen = image_datagen.flow(img, seed = seed, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(mask, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)
    
    # fits the model on batches with real-time data augmentation:
#     model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                         steps_per_epoch=len(x_train) / 32, epochs=epochs)
        
    return train_gen


def val_gen_aug(img_folder, mask_folder, batch_size):
    
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    
    img = np.zeros((len(n), 256, 256, 5)).astype(np.float32)
    mask = np.zeros((len(n), 256, 256, 2), dtype=np.float32)

    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        val_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        val_img = np.zeros((256,256,5))
        #resize
        for a in range(5):
            val_img[:,:,a] = cv2.resize(val_img_0[:,:,a], (256, 256))
        img[i] = val_img #add to array - img[0], img[1], and so on.
        
        #val_mask
        val_mask_0 = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0
        val_mask = np.where(val_mask_0==2.0, 0.0, val_mask_0) 
        #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
        val_mask = cv2.resize(val_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
        mask[i,:,:,0] = np.squeeze(val_mask)
        mask[i,:,:,1] = np.squeeze(1-val_mask)
        
    #print(mask.shape)
#     mask = mask.reshape(len(n),256*256, 2)

    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
#     img_datagen.fit(img)
#     mask_datagen.fit(mask)
    
    img_gen = img_datagen.flow(img, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(mask, batch_size=batch_size, shuffle=True)
    val_gen = zip(img_gen, mask_gen)
    # fits the model on batches with real-time data augmentation:
#     model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                         steps_per_epoch=len(x_train) / 32, epochs=epochs)
        
    return val_gen