import os
import model
from utils import *
import numpy as np
import random
import re
from shutil import copyfile

def cal_mean(arr, i, j, w, h):
    num = 0
    tot = 0
    for l in range(1,5):
        for m in range(i-l, i+l):
            if (m>=0 and m<w):
                for n in range(j-l, j+l):
                    if (n>=0 and n<h and arr[m,n]>=0):
                        num+=1
                        tot+=arr[m,n]
                        if num>2:
                            return tot/num

def inter_neg(arr):
    print('arr')
    nega = np.where(arr<0)
    print('start',nega[0].shape)
    
    l = 4
    w,h = arr.shape
    for i in range(w):
        for j in range(h):
            if (arr[i,j]<0):
                arr[i,j] = cal_mean(arr,i,j,w,h)
    nega = np.where(arr<0)
    print('end',nega[0].shape)            
    return arr

                        


def load_data_multi(img_folder, save_folder, shape=128):
    n = os.listdir(img_folder)
    n.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
#     img = np.zeros((len(n), shape, shape, 6)).astype(np.float32)

    for i in range(len(n)): #initially from 0 to 16, c = 0. 
        print(n[i])
        
        train_img = np.load(img_folder+'/'+n[i]) #normalization:the range is about -100 to 360
        if(train_img.shape!=(shape,shape,6)):
            continue
            
        #mclean_roi_slope
        train_img[:,:,0] = inter_neg(train_img[:,:,0])
        train_img[:,:,0] = train_img[:,:,0] / 88
 
        #mclean_roi_aspect
        train_img[:,:,1] = (train_img[:,:,1]) / 360
 
        #mclean_roi_rough
        print('2')
        train_img[:,:,2] = inter_neg(train_img[:,:,2])
        train_img[:,:,2] = train_img[:,:,2] / 313
#         train_img[:,:,2] = np.where(train_img[:,:,2]<0, -1, train_img[:,:,2])
        
        #mclean_roi_tpi
        print('3')
#         train_img[:,:,3] = (train_img[:,:,3]+9999) / (9999+275)
        train_img[:,:,3] = inter_neg(train_img[:,:,3])
        train_img[:,:,3] = train_img[:,:,3] / 275
 
        #mclean_roi_tri
        
        train_img[:,:,4] = inter_neg(train_img[:,:,4])
        train_img[:,:,4] = train_img[:,:,4] / 305
#         train_img[:,:,4] = np.where(train_img[:,:,4]<0, -1, train_img[:,:,4])
        #mclean_roi
        train_img[:,:,-1] = inter_neg(train_img[:,:,-1])
        train_img[:,:,-1] = (train_img[:,:,-1]-527.82) / 1174
#         train_img[:,:,-1] = np.where(train_img[:,:,-1]<0, -1, train_img[:,:,-1])
        
            
#         img[i] = train_img
        np.save(os.path.join(save_folder,n[i]), train_img)

            
#     return img
    

def main():
    train_frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b/'
    save_frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b_norm/'
    mkdir(save_frame_path)
    #To save frame in all_frames_5m6b_norm after normalization
    load_data_multi(train_frame_path, save_frame_path, 128)
    
    
def save_mask():
    frame_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_frames_5m6b_norm/'
    mask_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_5m6b/'
    save_mask_path = '/home/yifanc3/dataset/data/selected_128_overlap/all_masks_5m6b_norm/'
    mkdir(save_mask_path)
    
    n = os.listdir(frame_path)
    n.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for i in range(len(n)):
        print(i)
        src = os.path.join(mask_path, n[i])
        dst = os.path.join(save_mask_path, n[i])
        copyfile(src, dst)
        
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
#     save_mask()
        