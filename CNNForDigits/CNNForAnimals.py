import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils
from keras import backend as K

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import time

def LoadMNISTImages(folder, testOrTraining):

    mndata = MNIST(folder)

    if (testOrTraining == 'TRAINING'):
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()

    return mndata, images, labels

def TrainModel(Xtrain, ytrain, number_of_hidden_layers, number_of_epochs):
    model = TrainModelOptimizer(Xtrain, ytrain, number_of_hidden_layers, number_of_epochs, tf.train.AdamOptimizer())
    return model

def TrainModelOptimizer(Xtrain, ytrain, number_of_hidden_layers, number_of_epochs, optimizer):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(number_of_hidden_layers, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(Xtrain, ytrain, epochs=number_of_epochs)   

    return model

def plotGridResults(n_epochs, n_hiddenlayers, accuracy, filename):
    accuracy = np.array(accuracy)
    accuracy=accuracy.reshape(len(n_epochs), len(n_hiddenlayers))
    fig2, ax2 = plt.subplots(figsize=(12,8))
    c=ax2.contourf(n_epochs,n_hiddenlayers,accuracy)
    ax2.set_xlabel('Number of epochs')
    ax2.set_ylabel('Number of hidden layers')
    fig2.colorbar(c)
    fig2.savefig(filename)

# =====================================================================

if __name__ == '__main__':

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   
    testMultipleParams = False

    testOptimizationFrameworks = True

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

    if testMultipleParams:

        # grid search for a suitable parameter for the hidden layers and epochs
        number_of_hidden_layers = np.array([10, 20]) # 30, 40, 50, 100, 150, 200])
        number_of_epochs = np.array([1, 2]) #, 3, 4, 5, 10, 20, 30])

        losses = list()
        accuracies = list()
        timings  = list()

        text_file = open("Output_new.txt", "w")

        # for the text file - print number of layers first - as we are making a table
        text_file.write('Number of hidden layers: ')
        for j in number_of_hidden_layers:
            text_file.write('%iTiming(s) Loss Accuracy ' % j)
                    
        for i in number_of_epochs:        
            text_file.write('\nNumber of epochs:%i ' % i)

            for j in number_of_hidden_layers:
                print('Number of epochs:', i) 
                print('Number of hidden layers:', j)           

                # time training 
                time1 = time.time()
                model = TrainModel(X_train, y_train, j, i)
                time2 = time.time()
                time_span = time2-time1
                timings.append(time_span)
                print('Training the model took: %s seconds' % time_span)
                text_file.write("%.3f " % time_span)

                # check accuracy
                test_loss, test_acc = model.evaluate(X_test, y_test)
                losses.append(test_loss)
                text_file.write("%.3f " % test_loss)
                accuracies.append(test_acc)
                text_file.write("%.3f " % test_acc)

                print('Test accuracy:', test_acc)
    
        text_file.close()

        plotGridResults (number_of_epochs, number_of_hidden_layers, accuracies, 'GridSearch_2.png')

        plotGridResults (number_of_epochs, number_of_hidden_layers, timings, 'GridSearch_2_timings.png')
         
    if testOptimizationFrameworks:
                
         losses1 = list()
         accuracies1 = list()
         timings1  = list()

         text_file = open("Output_optimizer.txt", "w")

         # Adam Optimizer
         text_file.write('Optimizer: Adam optimizer ')         
         time1 = time.time()
         model = TrainModelOptimizer(X_train, y_train, 128, 5, tf.train.AdamOptimizer())
         time2 = time.time()
         time_span = time2 - time1
         text_file.write('%.3f sec ' % time_span)         
                 
         # check accuracy
         test_loss, test_acc = model.evaluate(X_test, y_test)         
         print('Test accuracy: ', test_acc)
         text_file.write("%.3f\n" % test_acc)
                  
         # SGD Optimizer
         text_file.write('Optimizer: SGD')         
         learning_rate = 0.15
         optimizerSGD = tf.train.sgd(learning_rate);

         time1 = time.time()
         model = TrainModelOptimizer(X_train, y_train, 128, 5, optimizerSGD)
         time2 = time.time()
         time_span = time2 - time1
         text_file.write('%.3f sec ' % time_span)         

         # check accuracy
         test_loss, test_acc = model.evaluate(X_test, y_test)       
         print('Test accuracy: ', test_acc)
         text_file.write("%.3f\n" % test_acc)

         


         #text_file.close()