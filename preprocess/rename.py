import os 
from libtiff import TIFF
import numpy as np

# Function to rename multiple files 
def main(): 
    directory = '/home/yifanc3/dataset/data/origin/DEM_fill/all_frames_5m6b_fill7b'
    PATH = "/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b_fill7b"
    check_path = "/home/yifanc3/dataset/data/selected_128_overlap/all_masks_5m6b"
    i = 0
    for filename in os.listdir(check_path): 
        tif = TIFF.open(os.path.join(directory, 'DEM_fill_merge_'+filename[:-3]+'tif'))
        img = tif.read_image()
        dst = os.path.join(PATH,filename[-11:-3]+'npy')
        print(dst)
        np.save(dst,img)
        i += 1
            
def select_frames_as_masks():
    PATH = "/home/yifanc3/dataset/data/selected_128_overlap/"
    mask_path = PATH + 'all_masks_5m6b'
    frame_path = PATH + 'all_frames_5m6b_fill7b'
    masks = os.listdir(mask_path)
    frames = os.listdir(frame_path)
    
    for frame in frames:
        if not frame in masks:
            print(frame)
            file = os.path.join(frame_path, frame)
            os.remove(file)
            
def mv_frames():
    PATH = "/home/yifanc3/dataset/data/"
    NEW_FRAME = PATH + 'masks_5m_256overlap'
    SAVE_FRAME = PATH + 'selected_256_overlap/all_masks_5m'
    if not os.path.isdir(SAVE_FRAME):
        os.makedirs(SAVE_FRAME)
        
    FRAME_PATH = PATH + 'selected_256_overlap/all_masks_10m'
    all_frames = os.listdir(FRAME_PATH)
    for frame in all_frames:
        src = os.path.join(NEW_FRAME, frame[:-3]+'tif')
        tif = TIFF.open(src)
        img = tif.read_image() # 0 means annotated
        dst = os.path.join(SAVE_FRAME, frame)
        print(src, dst)
        np.save(dst,img)

    
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
#     mv_frames()
    #select_frames_as_masks()
