import matplotlib.pyplot as plt
import os

def summarize_performance(history, Model_path):
    print(history.history.keys())
    
    if('lr' in history.history.keys()):
        plt.plot(history.history['lr'])
        plt.title('Model lr')
        plt.ylabel('lr')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(Model_path,'lr.png'))
        plt.clf()
        plt.cla()
        plt.close()
         
    # Plot training & validation accuracy values
    plt.plot(history.history['iou_label'])
    plt.plot(history.history['val_iou_label'])
    plt.title('Model iou_label')
    plt.ylabel('iou_label')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(Model_path,'iou_label.png'))
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.plot(history.history['per_pixel_acc'])
    plt.plot(history.history['val_per_pixel_acc'])
    plt.title('Model per_pixel_acc')
    plt.ylabel('per_pixel_acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(Model_path,'per_pixel_acc.png'))
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.plot(history.history['Mean_IOU'])
    plt.plot(history.history['val_Mean_IOU'])
    plt.title('Model Mean_IOU')
    plt.ylabel('Mean_IOU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(Model_path,'Mean_IOU.png'))
    plt.clf()
    plt.cla()
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(Model_path,'loss.png'))
    plt.clf()
    plt.cla()
    plt.close()
    
    

