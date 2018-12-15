import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import random
import time

import sys
sys.path.append("../Common/")
from Data import DataReader

# =====================================================================

size = 128
batch_size = 32
hidden_layers = 200
epochs = 20

def TrainModel(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs):
    model = TrainModelOptimizer(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs, tf.train.AdamOptimizer())
    return model

def TrainModelOptimizer(train_generator, validation_generator, number_of_hidden_layers, number_of_epochs, optimizer):
    
    model = keras.Sequential([        
        keras.layers.Conv2D(64, kernel_size = 3, activation='relu', data_format='channels_last', input_shape=(size, size, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),        
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
        
    model.summary()

    model.compile( optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4), 
              #  optimizer=optimizer, 
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

# =====================================================================

if __name__ == '__main__':
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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
    X_train = train_x[:2520]
    y_train = train_y[:2520]
    X_valid = train_x[2520:3240:]
    y_valid = train_y[2520:3240:]
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
    X_train = X_train.reshape(2520, size, size, 1)
    X_valid = X_valid.reshape(720, size, size, 1)
    X_test = X_test.reshape(360, size, size, 1)

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

    model = TrainModel(train_generator, validation_generator, hidden_layers, epochs)

    results = model.evaluate_generator(test_generator, steps=1000)  
 
   