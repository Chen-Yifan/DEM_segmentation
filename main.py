# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:59 2019

@author: Shiru

modified by Yifan
"""

import os
from options.train_options import TrainOptions
from data_loader import load_feature_data, preprocess
from sklearn.model_selection import train_test_split
from define_model import define_model
from util.util import *

if __name__ == '__main__':
    
    # parse my options
    opt = TrainOptions().parse()
    print('point0, option parser finished')

    ''' 1. load_data, only select feature images, 
        default: load the gradient of DEM
        return:  min, max among all the input images
    '''
    frame_data, mask_data, minn, max = load_feature_data(opt.frame_path, opt.mask_path, gradient=True)

    '''2. split train_val_test:
            input_train/val/test
            label_train/val/test  '''
    input_train, input_test, label_train, label_test = train_test_split(
                frame_data, mask_data, test_size=0.15, shuffle=False)

    input_train, input_val, label_train, label_val = train_test_split(
                input_data, label_data, test_size=0.1, shuffle=False)
    
    n_train, n_test, n_val = len(input_train), len(input_test), len(input_val)
    print('***** #train: #test: #val = %d : %d :%d ******'%(n_train, n_test, n_val))
    
    Data_dict = {
        'train':[input_train.astype('float32'),
                 label_train.astype('uint8')],
        'val':[input_val.astype('float32'),
                label_val.astype('uint8')],
        'test':[input_test.astype('float32'),
                label_test.astype('uint8')]
        }
    
    '''3. preprocess_data
       -----
       normalize all the data 
    '''
    preprocess(Data_dict, minn, maxx, opt.input_shape)
    
    # the actual model
    mkdir(opt.result_path)
    mkdir(opt.model_path)
    m = define_model(Data, opt)
    
    
