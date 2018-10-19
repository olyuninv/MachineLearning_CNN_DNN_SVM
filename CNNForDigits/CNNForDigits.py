import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

def LoadMNISTImages(folder, testOrTraining):

    mndata = MNIST(folder)

    if (testOrTraining == 'TRAINING'):
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()

    return mndata, images, labels

def TrainModel(Xtrain, ytrain, number_of_hidden_layers, number_of_epochs):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(number_of_hidden_layers, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(Xtrain, ytrain, epochs=number_of_epochs)   

    return model

def plotGridResults(n_epochs, n_hiddenlayers, accuracy):
    accuracy = np.array(accuracy)
    accuracy=accuracy.reshape(len(n_epochs), len(n_hiddenlayers))
    fig2, ax2 = plt.subplots(figsize=(12,8))
    c=ax2.contourf(n_epochs,n_hiddenlayers,accuracy)
    ax2.set_xlabel('Number of epochs')
    ax2.set_ylabel('Number of hidden layers')
    fig2.colorbar(c)
    fig2.savefig('GridSearch_2.png')

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

    class_names = ['0', '1', '2', '3', '4', 
               '5', '6', '7', '8', '9']

    #plt.ion()
    plt.figure()
    plt.imshow(X_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.savefig('firstimageScale.png')
    
    fig = plt.figure()
    for i in range(16):
      plt.subplot(4,4,i+1)
      plt.tight_layout()
      plt.imshow(X_train[i], cmap='gray', interpolation='none')
      plt.title("Digit: {}".format(y_train[i]))
      plt.xticks([])
      plt.yticks([])
    fig
    plt.savefig('exampleImages.png')

    #normalise data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    #num_classes = 10
    #y_train = np_utils.to_categorical(y_train, num_classes)
    #y_test = np_utils.to_categorical(y_test, num_classes)

    model = TrainModel(X_train, y_train, 128, 5)

    # check accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    predictions = model.predict(X_test)    
    likeliestPred = np.argmax(predictions[0])   #most likely prediction
    print('First item is most likely:', np.argmax(predictions[0]))

    # grid search for a suitable parameter for the hidden layers and epochs
    number_of_hidden_layers = np.array([10, 20, 30, 40, 50, 100, 150, 200])
    number_of_epochs = np.array([1, 2, 3, 4, 5, 10, 20, 30])

    losses = list()
    accuracies = list()

    for i in number_of_epochs:
        for j in number_of_hidden_layers:
            print('Number of epochs:', i)
            print('Number of hidden layers:', j)
            model = TrainModel(X_train, y_train, j, i)
        
            # check accuracy
            test_loss, test_acc = model.evaluate(X_test, y_test)
            losses.append(test_loss)
            accuracies.append(test_acc)
            print('Test accuracy:', test_acc)
    
    plotGridResults (number_of_epochs, number_of_hidden_layers, accuracies)

    #train_datagen = ImageDataGenerator(
    #    rescale=1./255, 
    #    rotation_range=40, 
    #    width_shift_range=0.2, 
    #    height_shift_range=0.2, 
    #    shear_range=0.2, 
    #    zoom_range=0.2, 
    #    horizontal_flip=True, 
    #    fill_mode='nearest') 

    #train_datagen.fit(X_train)
    
    #epochs = 1

    #print("train_generator");
    #train_generator = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
    #                steps_per_epoch=len(X_train) / 32, epochs=epochs)  