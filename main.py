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
from define_model import define_model, test_model
from visualize import visualize
from util.util import *
import sys
import csv


def train_test_val_split(frame_data,mask_data, opt):
    n = len(frame_data)
    a = int(n*0.75)
    b = int(n*0.85)
    # record test_names list in a csv
    names = os.listdir(opt.frame_path)
    test_names = names[b:]
    with open(opt.result_path+'/test_names.csv', 'w') as myfile:
        wr = csv.writer(myfile, dialect='excel')
        wr.writerow(test_names)
    x_train,x_val,x_test = frame_data[:a],frame_data[a:b],frame_data[b:]
    y_train,y_val,y_test = mask_data[:a],mask_data[a:b],mask_data[b:]
    return x_train, x_val, x_test, y_train, y_val, y_test

def main():

    opt = TrainOptions().parse()
    print('point0, option parser finished')
    
    if opt.isTrain:
        ''' 1. load_data, only select feature images, 
            default: load the gradient of DEM
            return:  min, max among all the input images
        '''

        frame_data, mask_data, minn, maxx = load_feature_data(opt.frame_path, opt.mask_path, 
                                                              gradient=False,dim=opt.input_shape,shuffle=False)
        print(np.min(frame_data),np.max(frame_data),np.unique(mask_data))
        print('point1, finish loading data')

        print('point2, shape frame mask', frame_data.shape, mask_data.shape)
        '''2. split train_val_test:
                input_train/val/test
                label_train/val/test  '''
        
        input_train, input_val, input_test, label_train, label_val, label_test = train_test_val_split(frame_data,mask_data,opt)
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

        '''3. preprocess_data
           -----
           normalize all the data 
        '''
        preprocess(Data_dict, minn, maxx, opt.input_shape)
    
        # the actual model
        mkdir(opt.model_path)
        img, real, pred = define_model(Data_dict, opt)
        
    else:
        # test/ prediction
        print('===========test==========')
        img, real, pred = test_model(opt)
        
    # visualize result
#     img = np.load(opt.result_path + '/inputs.npy')    
#     real = np.load(opt.result_path + '/gt_labels.npy')    
    
    
    result_dir = opt.result_path + '/epoch%s/'%opt.n_epoch
    for i in range(100):
        visualize(result_dir,img,real,pred,i)
        
    
main()    
