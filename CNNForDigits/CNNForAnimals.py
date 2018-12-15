#import tensorflow as tf
#from tensorflow import keras
#from keras.datasets import mnist
#from keras.preprocessing.image import ImageDataGenerator 
#from keras.utils import np_utils
#from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import random
import time

import sys
sys.path.append("../Common/")

from Data import DataReader

# =====================================================================

if __name__ == '__main__':
    dataLion = DataReader.readImages('../../Data/n02118333/', 0, 200, True)
    dataFox = DataReader.readImages('../../Data/n02129165/', 1, 200, True)
    dataTortoise = DataReader.readImages('../../Data/n01669191/', 2, 200, True)

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