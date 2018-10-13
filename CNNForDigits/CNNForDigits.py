import keras
from keras.datasets import mnist

from mnist import MNIST
import matplotlib.pyplot as plt

def LoadMNISTImages(folder, testOrTraining):

    mndata = MNIST(folder)

    if (testOrTraining == 'TRAINING'):
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()

    return mndata, images, labels

# =====================================================================

if __name__ == '__main__':
# Download the data set from URL
    #print("Loading data")
    #mndata, images, labels = LoadMNISTImages('./Data', 'TRAINING')

    #print("Loading test data")
    #mndata, images_test, labels_test = LoadMNISTImages('./Data', 'TEST')
    
    #print(mndata.display(images[0]))
    #print(mndata.display(images_test[0]))
    
    #load mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    fig = plt.figure()
    for i in range(9):
      plt.subplot(3,3,i+1)
      plt.tight_layout()
      plt.imshow(X_train[i], cmap='gray', interpolation='none')
      plt.title("Digit: {}".format(y_train[i]))
      plt.xticks([])
      plt.yticks([])
    fig
