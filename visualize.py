

predicted_data = np.zeros(pred.shape)

for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        for k in range(pred.shape[2]):
            if (pred[i,j,k]>=0.3):
                predicted_data[i,j,k] =1
            else:
                predicted_data[i,j,k] =0

for i in range(100):
    visualize(real,pred,predicted_data,i)