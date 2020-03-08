import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import  Callback

# learning rate scheduler for classifier
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 60:
        lrate = 0.0001
    elif epoch > 70:
        lrate = 0.00001
    return lrate

def GetCifar10Mean():
    return np.array([0.491, 0.482, 0.446], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]

def GetCifar10STD():
    return np.array([0.247, 0.243, 0.261], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]

def GetImageNetMean():
    return np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]

def GetImageNetSTD():
    return np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B] 


######## Utils for Autoencoders #########
class SaveOutputImages(Callback):
    '''
      Save original and reconstructed images after each epoch during training process. 
    '''
    def __init__(self, validation_generator, save_dir):
        self.x_test = validation_generator.next()
        self.out_dir = save_dir
        self.count = 0

    def on_epoch_end(self, batch, logs={}):
        self.count+=1
        decoded_imgs = self.model.predict(self.x_test)

        n = 10
        plt.figure(figsize=(n*2, 2*2))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i+1)
            plt.imshow(self.x_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n+1)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(self.out_dir + "/" + "ae_ep%d.jpg"%(self.count),  bbox_inches='tight', dpi=100)
        plt.close('all')


def VisualizeAE(original_imgs, decoded_imgs , out_dir, epochs):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(out_dir + "/ae_trained_ep%d.jpg"%(epochs), bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()


def PlotTrainValAccuracy(history, out_dir, epochs):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(out_dir + "/" + "train_val_accuracy_ep%d.jpg"%(epochs), bbox_inches='tight', dpi=150)
    #plt.show()
    plt.close()

def PlotTrainValLoss(history, out_dir, epochs):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(out_dir + "/" + "train_val_loss_ep%d.jpg"%(epochs), bbox_inches='tight', dpi=150)
    #plt.show()
    plt.close()


def CalculateConfusionMatrix(gt_list, pred_list, target_names, img_out_path, accuracy):
    print("Calculating confusion matrix ...")
    cm = confusion_matrix(y_true=gt_list, y_pred=pred_list)
    # normalized confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    num_classes = len(target_names)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for CIFAR10. Accuracy: %f'%(accuracy))

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
              ha="center", va="center",
              color="white" if cm[i, j] > thresh else "black")

    figure = plt.gcf() 
    figure.set_size_inches(15, 15)

    plt.savefig(img_out_path, bbox_inches='tight', dpi = 200)
    #plt.show()
    plt.close()
