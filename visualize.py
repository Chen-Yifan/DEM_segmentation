from matplotlib import pyplot as plt

def visualize(result_path, real_data, predict_data, predicted_data, idx):
    
    f = plt.figure(figsize = (10,5))
    
    f.add_subplot(1,3,1)
    cs = plt.imshow(real_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('real')
    
    f.add_subplot(1,3,2)
    cs = plt.imshow(predict_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('predicted')
    
    # predicted_data = np.zeros((predict_data.shape[0], predict_data.shape[1], predict_data.shape[2]))
    # f.add_subplot(1,3,3)
    # for i in range(predict_data.shape[0]):
        # for j in range(predict_data.shape[1]):
            # for k in range(predict_data.shape[2]):
                # if (predict_data[i,j,k]>=0.3):
                    # predicted_data[i,j,k] =1
                # else:
                    # predicted_data[i,j,k] =0
    f.add_subplot(1,3,3)
    cs = plt.imshow(predicted_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('pred_processed')	
    plt.show()
    plt.savefig(result_path+'/mask_comparison_idx%d.png'%(idx))
    plt.close()