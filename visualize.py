import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def process_comp(gt, pred):
    """
    Process groundtruth label and predicted label to 3 class map,
    where TP=1, FP=2, FN=3, TN=0.
    
    Arugments:
        gt: binary numpy array (n, dim,dim,#channel)
        pred: binary numpy array (n, dim,dim,#channel)
        
    Return:
        comparison array named comp_arr of same dimension with gt and pred.
    """
    comp_arr = np.zeros(gt.shape)
    comp_arr = np.zeros(gt.shape)
    comp_arr[np.multiply((gt==1),(pred==1))]=1
    comp_arr[np.multiply((gt==0),(pred==1))]=2
    comp_arr[np.multiply((gt==1),(pred==0))]=3
    return comp_arr
    

def map_comparr_color(arr):
    """
    Map the 3-class comparison array from Func:process_comp to different colors.
    
    Arguments:
        arr: 3-class numpy array (n, dim, dim, 1), where TP=1, FP=2, FN=3, TN=0.
        green for TP=1, Red for FP=2, yellow for FN=3, TN are transparent
        
    Return:
        rgb colorful array in (n, dim, dim, 3) type is uint8.
    """
    r=(255,0,0)
    g=(0,255,0)
    y=(255,255,0)
    arr_3channel = np.zeros((arr.shape[0],arr.shape[1],arr.shape[2], 3))
    arr_3channel[np.where(arr==1)]= g
    arr_3channel[np.where(arr==2)]= r
    arr_3channel[np.where(arr==3)]= y
    
    #TODO: add background layer as transparent
    
    return arr_3channel.astype('uint8')

def fourplot(img1, img2, img3, img4, idx, save_path):
    
    f = plt.figure(figsize = (10,10))
    # f.suptitle('The'+str(idx)+'th image')
    
    f.add_subplot(2,2,1)
    cs = plt.imshow(img1)
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('input')
    
    f.add_subplot(2,2,2)
    cs = plt.imshow(img2)
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('comparison')
    
    f.add_subplot(2,2,3)
    cs = plt.imshow(img3)
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('groundtruth')
    
    f.add_subplot(2,2,4)
    cs = plt.imshow(img4)
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('prediction')	
    plt.show()
    plt.savefig(save_path+'/4plot_idx%d.png'%(idx))
    plt.close()


def save_visualization(result_path, epoch, threshold):
    """
    Save visualization of results in the directory indicated by result_path.
    The visualization will be a 2x2 matplotlib, with the NW: DEM, NE:3comp_arr, SW: groundtruth, SE:prediction
    
    Arguments:
        result_path: containing epoch**/pred_labels.npy  gt_labels.npy  inputs.npy  test_names.csv
        epoch: int; indicate from which epoch the prediction comes
        threshold: float, threshold to place on pred_labels.npy

    """
    epoch_path = result_path+'/epoch%d'%epoch
    X = np.squeeze(np.load(result_path+'/inputs.npy'),axis=-1)                  # load DEM input array (n, 128, 128)
    Y_gt = np.squeeze(np.load(result_path+'/gt_labels.npy'),axis=-1)            # load binary groundtruth label array (n, 128, 128)
    Y_pred = np.load(epoch_path+'/pred_labels.npy')        # load predicted array, float (n, 128, 128, #channel)
    # place threshold and convert to a binary array (n,128,128)
    if Y_pred.shape[-1]==1:
        Y_pred = np.greater(np.squeeze(Y_pred,axis=-1),threshold).astype('uint8')   
    else:
        Y_pred = np.argmax(Y_pred,axis=-1)
        
    """ Process to generate a 3-color rgb array """
    comp_arr = process_comp(Y_gt, Y_pred)
    color_arr = map_comparr_color(comp_arr)   # 3-color rgb array in (n,128,128,3) type:uint8
    
    
    """Plot 4 images on one figure and save """
    for i in range(len(X)):
        fourplot(X[i], color_arr[i], Y_gt[i], Y_pred[i], i, epoch_path)         # save plot
        
        color_im = Image.fromarray(color_arr[i]) 
        color_im.save(epoch_path+"/comp%d.png"%(i))                             # save color image
        
        if i>100:
            break
