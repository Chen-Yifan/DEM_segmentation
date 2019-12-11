# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:38:59 2019

@author: fffibonacci

Equipped the spatial datagenerator to do normalization, select corresponding pairs

"""
import pytest
from keras_spatial.datagen import SpatialDataGenerator
import keras_spatial.grid as grid
from geopandas import GeoDataFrame
from rasterio.crs import CRS
import numpy as np
from dataGenerator import *
def test_preprocess_modify_array():

    def pre(arr, maxval):
        return arr / maxval

    size = (64,64)
    sdg = SpatialDataGenerator()
    sdg.source = path
    sdg.indexes = 1
    sdg.width, sdg.height = size
    df = sdg.regular_grid(*size)
    df['max'] = [a.max() for a in sdg.flow_from_dataframe(df, batch_size=1)]

    sdg.add_preprocess_callback('normalize', pre, df['max'].max())
    arr = next(sdg.flow_from_dataframe(df))
    assert len(arr.shape) == 3
    assert arr.shape[0] == min(sdg.batch_size, len(df))
    assert arr.shape[-2] == size[0] and arr.shape[-1] == size[1]
    assert arr.max() <= 1.0
    print('test_preprocess_modify_array() passed')
    
def dataGen(new_batch_size=16):
    
    ''' SpatialDataGenerator for frame and masks 
        from repo: https://github.com/IllinoisStateGeologicalSurvey/keras-spatial
        
        args:
            DEM_path, label_path: are DEM and label file respectively
            sdg: the input data generator
                we normalize it by its globle max and min 
                Note: many -9999, we only normalize positive values
               
            sdg2:  the output data generator
                
            filter_to_array: function to filter the generator and return x,y arrays
    '''
    def pre(arr):
        return 1 - np.where(arr==2,0,1)
    
    def norm(arr, maxval, minval):
        return (arr - minval) / (maxval - minval)
    
    DEM_path = '/home/yifanc3/dataset/data/origin/DEM_fill/DEM_fill_no_data.tif'
    label_path = '/home/yifanc3/dataset/data/origin/cl1.tif'
    size = (128,128)
    batch_size = 1
    
    sdg = SpatialDataGenerator()
    sdg.interleave = 'pixel'
    sdg.source = DEM_path
    sdg.width, sdg.height = size
    sdg.batch_size = batch_size
    df = sdg.regular_grid(*size)
    # clear negative values + normalization
#     df['max'] = [a.max() for a in sdg.flow_from_dataframe(df, batch_size=1)]
#     df['min'] = [a[np.where(a>=0)].min() for a in sdg.flow_from_dataframe(df, batch_size=1)]
#     print('global max/min (ignore negative values)',df['max'].max(), df['min'].min())
#     sdg.add_preprocess_callback('normalize', norm, df['max'].max(), df['min'].min())
    
    sdg2 = SpatialDataGenerator(batch_size=batch_size)
    sdg2.profile = sdg.profile
    sdg2.source = label_path
    df2 = sdg2.regular_grid(*size)
    print('replace 1 to 0, 2 to 1, from now on 0 is background, 1 is feature')
    sdg2.add_preprocess_callback('label_0_1', pre)
    
    frameGen = sdg.flow_from_dataframe(df)
    maskGen = sdg2.flow_from_dataframe(df2)
    
    print('length of dfs',len(df), len(df2))
    gen = zip(frameGen, maskGen)
    print('dataGen to array')
    x,y = filter_to_array(gen,size,new_batch_size)
    
    # normalize x
    print('normalize it')
    minv = x[np.where(x>=0)].min()
    maxv = x.max()
    x = (x - minv)/(maxv-minv)
    return x,y


def filter_to_array(orig_gen, size, new_batch=16):
    '''
    filter both images and masks with no features
    and return x_array, y_array in shape [n, width, height, channel]
    '''
    x_array = []
    y_array = []
    i = 0
    while True:
        try:
            x, y = next(orig_gen)
        except:
            print('stopIterator')
            break
        if np.max(y)==0:
            continue
        i+=1
        x_array.append(x)
        y_array.append(y)
    x_array = np.squeeze(np.array(x_array),axis=1)
    y_array = np.squeeze(np.array(y_array),axis=1)
    return x_array,y_array
    
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    dataGen()
    
