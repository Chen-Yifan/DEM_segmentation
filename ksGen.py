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
                
            select_gen(original_gen, size, new_batch)
                we need to select those pairs with features only, eliminate those without features
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
    df['max'] = [a.max() for a in sdg.flow_from_dataframe(df, batch_size=1)]
    df['min'] = [a[np.where(a>=0)].min() for a in sdg.flow_from_dataframe(df, batch_size=1)]
    print('global max/min (ignore negative values)',df['max'].max(), df['min'].min())
    sdg.add_preprocess_callback('normalize', norm, df['max'].max(), df['min'].min())
    
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
    print('old dataGen to new dataGen')
    newGen = filter_gen(gen,size,new_batch_size)
    
    # start testing
    # x,y = next(newGen)
    # print('back',x.shape, y.shape, np.max(x), np.min(x), np.unique(y))
    return newGen
    
def filter_gen(orig_gen, size, new_batch=16):
    '''from old gen to new gen
    
    args:
        orig_gen: original generator with batch_size=1
        size: size of the generated image
            (default: channel = 1)
        new_batch: new batch size after the filter
        
    '''
    while True:
        i = 0
        batch_x = np.zeros((new_batch,size[0],size[1], 1))
        batch_y = np.zeros((new_batch,size[0],size[1], 1))
        
        while i < new_batch:
            x, y = next(orig_gen)
            batch_size = len(x)
            if np.max(y)==0:
                continue
            batch_x[i]=x
            batch_y[i]=y
            i+=1
        print('new_batch',batch_x.shape, batch_y.shape)
        if(len(batch_x.shape)!=new_batch):
            break
        yield (batch_x, batch_y)   
    
    
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    dataGen()
    
