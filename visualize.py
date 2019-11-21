import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

def visualize(result_path, img_data, real_data, predict_data, idx):
    
    if real_data.ndim==4:
        real_data = real_data[:,:,:,0]
    if img_data.ndim==4:
        img_data = img_data[:,:,:,0]
    if predict_data.shape[-1]==1:
        predict_data = predict_data[:,:,:,0]
        predicted_data = (predict_data>=0).astype('uint8')
        print('predicted_data',predicted_data.shape)
    elif predict_data.shape[-1]==2:
        predict_data = np.argmax(predict_data,axis=-1)
        predicted_data = predict_data

    f = plt.figure(figsize = (10,10))
    
    f.add_subplot(2,2,1)
    cs = plt.imshow(img_data[idx])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('image')
    
    f.add_subplot(2,2,2)
    cs = plt.imshow(real_data[idx])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('real')
    
    f.add_subplot(2,2,3)
    cs = plt.imshow(predict_data[idx])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('predicted')
    
    f.add_subplot(2,2,4)
    cs = plt.imshow(predicted_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('pred_processed')	
    plt.show()
    plt.savefig(result_path+'/mask_comparison_idx%d.png'%(idx))
    plt.close()
