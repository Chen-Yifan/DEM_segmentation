import pytest
from keras_spatial.datagen import SpatialDataGenerator
import keras_spatial.grid as grid
from geopandas import GeoDataFrame
from rasterio.crs import CRS
import numpy as np

def SDG(frame_path, mask_path, size=(128,128),overlap=0.5,batch_size=32):
    def pre(arr, maxval):
        return arr / maxval
    
    sdg = SpatialDataGenerator(source=frame_path, indexes=1,interleave = 'pixel')
    sdg.width, sdg.height = size
    df = sdg.regular_grid(*size, overlap)
    df['max'] = [a.max() for a in sdg.flow_from_dataframe(df, batch_size=1)]

    sdg.add_preprocess_callback('normalize', pre, df['max'].max())
    frame_gen = sdg.flow_from_dataframe(df)
    
    print('test_preprocess_modify_array() passed')
    
    sdg = SpatialDataGenerator(source=mask_path,indexes=1,interleave = 'pixel')
    sdg.width, sdg.height = size
    df2 = sdg.regular_grid(*size, overlap)

    mask_gen = sdg.flow_from_dataframe(df2)
    
    assert len(df) == len(df2)
    
    return frame_gen, mask_gen, len(df)



def test_preprocess_add_array():

    def pre(arr):
        return np.stack((arr, arr/10))

    size = (64,64)
    sdg = SpatialDataGenerator()
    sdg.source = 'data/small.tif'
    sdg.indexes = 1
    sdg.width, sdg.height = size
    df = sdg.regular_grid(*size)

    sdg.add_preprocess_callback('pre', pre)
    arr = next(sdg.flow_from_dataframe(df))
    assert len(arr.shape) == 4
    assert arr.shape[0] == min(sdg.batch_size, len(df))
    assert arr.shape[1] == 2
    assert arr.shape[-2] == size[0] and arr.shape[-1] == size[1]


    
