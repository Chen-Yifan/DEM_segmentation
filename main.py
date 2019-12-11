# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:59 2019

@author: Shiru

modified by Yifan
"""

import os
import numpy as np
from options.train_options import TrainOptions
from data_loader import load_feature_data, preprocess
from sklearn.model_selection import train_test_split
from define_model import define_model, test_model
from visualize import visualize
from util.util import *
import sys
from ksGen import dataGen


def main():

    opt = TrainOptions().parse()
    print('point0, option parser finished')

    ''' 1. load_data, only select feature images, 
        default: load the gradient of DEM
        return:  min, max among all the input images
    '''
    print('point1, finished load data')
    frame_data, mask_data = dataGen(opt.batch_size)
    
    print('point2, shape frame mask', frame_data.shape, mask_data.shape)
    '''2. split train_val_test:
            input_train/val/test
            label_train/val/test  '''
    input_train, input_test, label_train, label_test = train_test_split(
                frame_data, mask_data, test_size=0.15, shuffle=False)

    input_train, input_val, label_train, label_val = train_test_split(
                frame_data, mask_data, test_size=0.1, shuffle=False)
    print('point3, shape frame mask', input_train.shape, label_train.shape)

    n_train, n_test, n_val = len(input_train), len(input_test), len(input_val)
    print('***** #train: #test: #val = %d : %d :%d ******'%(n_train, n_test, n_val))
    
    Data_dict = {
        'train':[input_train.astype('float32'),
                 label_train.astype('float32')],
        'val':[input_val.astype('float32'),
                label_val.astype('float32')],
        'test':[input_test.astype('float32'),
                label_test.astype('float32')]
    }
    
    mkdir(opt.result_path)
    if opt.isTrain:
        # the actual model
        mkdir(opt.model_path)
        define_model(Data_dict, opt)
    
    else:
        # test/ prediction
        print('===========test==========')
        test_model(Data_dict, opt)
        
    # visualize result
    img = Data_dict['test'][0][:,:,:,0]
    real = np.load(opt.result_path + '/gt_labels.npy')
    pred = np.load(opt.result_path + '/pred_labels.npy') 
    
    
    predicted_data = np.zeros(pred.shape)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                if (pred[i,j,k]>=0.5):
                    predicted_data[i,j,k] =1
                else:
                    predicted_data[i,j,k] =0
	
    for i in range(100):
        visualize(opt.result_path,img,real,pred,predicted_data,i)
        
    
main()    
