import numpy as np
import cv2
#from scipy.misc import imresize
import os
import random
from keras.utils import to_categorical

NUMBER_OF_CLASSES = 2
IMAGE_W = 256
IMAGE_H = 256

def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)

    while (True):
        img = np.zeros((batch_size, 256, 256, 5)).astype(np.float32)
        mask = np.zeros((batch_size, 256, 256, 2), dtype=np.float32)

        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            train_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
            #train_img =  cv2.resize(train_img, (256, 256))# Read an image from folder and resize
            train_img = np.zeros((256,256,5))
            #resize
            for a in range(5):
                train_img[:,:,a] = cv2.resize(train_img_0[:,:,a], (256, 256))


            img[i-c] = train_img #add to array - img[0], img[1], and so on.


            #train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            #train_mask = cv2.resize(train_mask, (256, 256))
            #train_mask = train_mask.reshape(256, 256, 1) # Add extra dimension for parity with train_img size [256 * 256 * 5]
            train_mask_0 = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0
            #change 2--0
            train_mask = np.where(train_mask_0==2.0, 0.0, train_mask_0) 
            # resize to 256, 256, 1
            #train_mask = imresize(train_mask[:,:,a], (256, 256), interp='nearest').astype('float32')
            train_mask = cv2.resize(train_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            #train_mask = train_mask.reshape(256, 256, 1)
           # train_mask = to_categorical(train_mask, 2)
            mask[i-c,:,:,0] = np.squeeze(train_mask)
            mask[i-c,:,:,1] = np.squeeze(1-train_mask)
#             mask[i-c] = train_mask
        #print(mask.shape)
        mask = mask.reshape(batch_size,IMAGE_H*IMAGE_W, 2)
        
        c+=batch_size
        if(c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n)
        yield img, mask


def test_gen(img_folder, mask_folder):

    n = os.listdir(img_folder) #List of training images
#     random.shuffle(n)


    img = np.zeros((len(n), 256, 256, 5)).astype(np.float32)
    mask = np.zeros((len(n), 256, 256, 2), dtype=np.int64)

    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        test_img_0 = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        #train_img =  cv2.resize(train_img, (256, 256))# Read an image from folder and resize
        test_img = np.zeros((256,256,5))
        #resize
        for a in range(5):
            test_img[:,:,a] = cv2.resize(test_img_0[:,:,a], (256, 256))

        img[i] = test_img #add to array - img[0], img[1], and so on.
        
        
        test_mask_0 = np.load(mask_folder+'/'+n[i]) # 1.0 or 2.0
#         #change 2--0
        test_mask = np.where(test_mask_0==2.0, 0.0, test_mask_0)
        # resize to 256, 256, 1
        test_mask = cv2.resize(test_mask,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int64)
        #test_mask = test_mask.reshape(256, 256, 1)
        #test_mask = to_categorical(test_mask, 2)
#        mask[i] = test_mask
        mask[i,:,:,0] = np.squeeze(test_mask)
        mask[i,:,:,1] = np.squeeze(1-test_mask)
    
    mask = mask.reshape(len(n),IMAGE_H*IMAGE_W,2)
        
    #mask = to_categorical(mask, 2) 
        
    return img, mask
