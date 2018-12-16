#import tensorflow as tf
#from tensorflow import keras
from keras import layers 
from keras import models 
from keras import optimizers 

from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils
from keras import backend as K
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import os
import time

import gc

import sys
sys.path.append("../Common/")
from Data import DataReader
from Graphs import Graphs

# =====================================================================

size = 128
batch_size = 32
hidden_layers = 200
epochs = 20

def TrainModel(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs):
    model = TrainModelOptimizer(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs, optimizers.Adam(lr=1e-4))
    return model 

def TrainModelOptimizer(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs, optimizer):
    
    model = models.Sequential() 
    model.add(layers.Conv2D(number_of_hidden_layers, kernel_size = 5, activation='relu', data_format='channels_last', input_shape=(size, size, 1), padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(int(number_of_hidden_layers/2), kernel_size=3, activation='relu', data_format='channels_last', input_shape=(size/2, size/2, 1), padding = 'same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten()),#input_shape=(size/4, size/4, 1)))
    model.add(layers.Dense(128, activation='relu'))        
    model.add(layers.Dense(3, activation='softmax'))
        
    model.summary()

    model.compile( 
              #optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4), 
              optimizer=optimizer, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit_generator( 
    train_generator, 
    steps_per_epoch=100, 
    epochs=number_of_epochs,  
    verbose=2, 
    validation_data=validation_generator, 
    validation_steps=50) 	

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
    
    loadData = True
    trainSingleNetwork = False
    testMultipleParams1 = False
    testMultipleParams2 = True
    plotResults = False

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if loadData:
        label_dict = {
            0: 'Lion',
            1: 'Fox',
            2: 'Tortoise'
        }

        dataLion = DataReader.readImages('../../Data/n02118333/', 0, size, True)
        dataFox = DataReader.readImages('../../Data/n02129165/', 1, size, True)
        dataTortoise = DataReader.readImages('../../Data/n01669191/', 2, size, True)

        # Keep first 1200
        dataLion = dataLion[:1200]
        dataFox = dataFox[:1200]
        dataTortoise = dataTortoise[:1200]

        dataset = np.concatenate([dataLion, dataFox, dataTortoise])
        np.random.shuffle(dataset)

        train_x = list()
        train_y = list()
    
        # split into x and y
        for (x, y) in dataset:
            train_x.append(x)
            train_y.append(y)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        print("Successfully loaded data")

        # split data into test train validation
        X_train = train_x[:2880]
        y_train = train_y[:2880]
        X_valid = train_x[2880:3240:]
        y_valid = train_y[2880:3240:]
        X_test = train_x[3240:]
        y_test = train_y[3240:]
    
        plt.figure()
        plt.imshow(X_train[0])
        plt.colorbar()
        plt.grid(False)
        plt.savefig('firstimageAnimal.png')
    
        fig = plt.figure()
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.tight_layout()
            plt.imshow(X_train[i], cmap='gray', interpolation='none')
            plt.title("Label: {}".format(y_train[i]))
            plt.xticks([])
            plt.yticks([])
        fig
        plt.savefig('exampleAnimalImages.png')

        # reshape into rank 4
        X_train = X_train.reshape(2880, size, size, 1)
        X_valid = X_valid.reshape(360, size, size, 1)
        X_test = X_test.reshape(360, size, size, 1)

        #one-hot encode target column
        #y_train = to_categorical(y_train)
        #y_valid = to_categorical(y_valid)
        #y_test = to_categorical(y_test)

        # create ImageDataGenerator to augment images    
        train_datagen = ImageDataGenerator(rescale=1./255,
                                            rotation_range=40, 
                                            width_shift_range=0.2, 
                                            height_shift_range=0.2, 
                                            shear_range=0.2, 
                                            zoom_range=0.2, 
                                            horizontal_flip=True, 
                                            fill_mode='nearest',
                                            data_format = "channels_last") 

        validation_datagen = ImageDataGenerator(rescale=1./255) 
        test_datagen = ImageDataGenerator(rescale=1./255) 
    
        # connect ImageDataGenerator and 
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
        validation_generator = validation_datagen.flow(X_valid, y_valid, batch_size=batch_size, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
        test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)

    if trainSingleNetwork:
        model = TrainModel(train_generator, validation_generator, 20, 5)
        results = model.evaluate_generator(test_generator, steps=1000)  
        print('Final test loss:', (results[0]*100.0))
        print('Final test accuracy:', (results[1]*100.0))

        fileName = "./models/model_%ilayers_%iepochs.h5" % (hidden_layers, epochs)
        
        exists = os.path.isfile(fileName)
        if exists:
            os.remove(fileName)
        
        model.save(fileName) 
    
    if testMultipleParams1:

        # grid search for a suitable parameter for the hidden layers and epochs
        number_of_hidden_layers = np.array([10, 20, 30, 40, 50, 100, 150, 200])
        number_of_epochs = np.array([30]) #, 1, 2, 3, 4, 5, 10, 20])

        losses = list()
        accuracies = list()
        timings  = list()

        text_file = open("Output_Animals.txt", "a") #"w")

        # for the text file - print number of layers first - as we are making a table
        #text_file.write('Number of hidden layers: ')
        #for j in number_of_hidden_layers:
            #text_file.write('%iTiming(s) Loss Accuracy ' % j)
                    
        for i in number_of_epochs:        
            text_file.write('\nNumber of epochs:%i ' % i)

            for j in number_of_hidden_layers:
                print('Number of epochs:', i) 
                print('Number of hidden layers:', j)           

                # time training 
                time1 = time.time()
                model = TrainModel(train_generator, validation_generator, j, i)
                time2 = time.time()
                time_span = time2-time1
                timings.append(time_span)
                print('Training the model took: %s seconds' % time_span)
                text_file.write("%.3f " % time_span)

                # check accuracy
                results = model.evaluate_generator(test_generator, steps=100) 
                test_loss = results[0] * 100
                test_acc = results[1] * 100
                losses.append(test_loss)
                text_file.write("%.3f " % test_loss)
                accuracies.append(test_acc)
                text_file.write("%.3f " % test_acc)

                print('Test accuracy:', test_acc)
                fileName = "./models/model_%ilayers_%iepochs.h5" % (j, i)
        
                exists = os.path.isfile(fileName)
                if exists:
                    os.remove(fileName)
        
                model.save(fileName) 
                gc.collect()
                    
        text_file.close()

        #plotGridResults (number_of_epochs, number_of_hidden_layers, accuracies, 'GridSearch_animals.png')

        #plotGridResults (number_of_epochs, number_of_hidden_layers, timings, 'GridSearch__animals_timings.png')
 
    if plotResults:
        Graphs.plotGridResults ('Number of hidden layers', 'Number of epochs', '../CNNForDigits/Output_Animals1.csv', 'accuracy')