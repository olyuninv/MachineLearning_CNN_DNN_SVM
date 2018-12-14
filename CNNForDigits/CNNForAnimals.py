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

import sys
sys.path.append("../Common/")

from Data import DataReader

# =====================================================================

if __name__ == '__main__':
    data = DataReader.readImages('../../Data/n02118333/', 200, True)
    
    printf("Successfully loaded data")