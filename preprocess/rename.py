import os 
from libtiff import TIFF
import numpy as np
  
# Function to rename multiple files 
def main(): 
    directories = ['frames_128_overlap6']
    PATH = "/home/yifanc3/dataset/data/"
    i = 0
    for d in directories:
        file = os.path.join(PATH, d)
        print(file)
        for filename in os.listdir(file): 
            dst = file+'/'+filename[-11:]
            src = file+'/'+filename
            print(src,dst)
            os.rename(src, dst) 
            i += 1

            
def mv_frames():
    PATH = "/home/yifanc3/dataset/data/"
    NEW_FRAME = PATH + 'frames_128_overlap6'
    SAVE_FRAME = PATH + 'selected_128_overlap/all_frames6'
    if not os.path.isdir(SAVE_FRAME):
        os.makedirs(SAVE_FRAME)
        
    FRAME_PATH = PATH + 'selected_128_overlap/all_frames'
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

