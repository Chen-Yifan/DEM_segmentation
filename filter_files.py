import numpy as np
from PIL import Image
import os


def filter_DEM():
    path = '/home/shared/dem/building_data/512_50p_over/DEM_retile_all/'
    frame_list = os.listdir(path)
    a = frame_list.index('mclean_fillnodata_107_015.tif')
    frame_list = frame_list[a:]
    new_path = '/home/shared/dem/building_data/512_50p_over/DEM_retile/'
    for file in frame_list:
        array = np.array(Image.open(path+file))
        if np.max(array)!=0:
            print(file)
            np.save(new_path+file[11:-3]+'npy',array)
            
def filter_label():
    '''
     TODO:
         1. rename DEM_retile files to only 001_001.npy style
         2. load labels_retile to new file and rename it 
    '''
    path = '/home/shared/dem/building_data/512_50p_over/'
    DEM_path = path + 'DEM_retile/'
    label_old_path = path + 'labels_retile_all/'
    label_new_path = path + 'labels_retile/'
    
    frame_list = os.listdir(DEM_path)
    for file in frame_list:
        file_new = file[-11:]
        #DEM
        DEM_old = DEM_path + file
        DEM_new = DEM_path + file_new
        print(DEM_new)
        os.rename(DEM_old, DEM_new)

        #label
        label = np.array(Image.open(label_old_path+'mclean_building_label_'+file_new[:-3]+'tif'))
        np.save(label_new_path+file_new, label)

        
filter_label()        