# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:59 2019

@author: Shiru, Yifan
"""

import os
import numpy as np
from options.train_options import TrainOptions
from data_loader import load_data
from define_model import train_model, test_model
from visualize import save_visualization
from util.util import *
import sys
import csv


def main():

    opt = TrainOptions().parse()
    
    if opt.isTrain:
        ''' Load_data
            Function:   only select feature images, 
                        Default: load DEM, do all the calculation (gradient/aspect/hillshade) on the fly
                        Normalize:  min, max among the single image
        '''
        print('Start loading data ...')
        Data_dict = load_data(opt)
        
        
        # the actual model
        mkdir(opt.model_path)
        img, real, pred = train_model(Data_dict, opt)
        
    else: # test/ prediction
        print('===========test==========')
        img, real, pred = test_model(opt)
        
        # visualize result
        if(opt.visualize):
            print('=============Save Visualization 100 Images ===================')
            save_visualization(opt.result_path, opt.n_epoch, opt.threshold)
        
main()    
