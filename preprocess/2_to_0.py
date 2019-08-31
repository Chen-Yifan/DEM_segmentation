import os 
from libtiff import TIFF
import numpy as np

# Function to rename multiple files 

def main(): 
    '''
    Mask: cl1.tif contains two value, 1 and 2, where 1 is feature, 2 is background
    This function changes 2 to 0 in each small sampled images
    
    '''
    i = 0
    PATH = "/home/yifanc3/dataset/tif/train_masks/train_masks/"
    for filename in os.listdir(PATH): 
        file = PATH+filename
        print(file)
        tif = TIFF.open(file)
        img_0 = tif.read_image()
        img = np.where(img_0==2.0, 0.0, img_0) 
        tif = TIFF.open(file, mode='w')
        tif.write_image(img)
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 