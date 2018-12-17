import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils
from keras import backend as K

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time
import math

from tempfile import TemporaryFile

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

#def run_train(session, train_x, train_y):
#  print('\nStart training')
#  session.run(init)
#  for epoch in range(10):
#    total_batch = int(train_x.shape[0] / batch_size)
#    for i in range(total_batch):
#      batch_x = train_x[i*batch_size:(i+1)*batch_size]
#      batch_y = train_y[i*batch_size:(i+1)*batch_size]
#      _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
#      if i % 50 == 0:
#        print('Epoch #%d step=%d cost=%f' % (epoch, i, c))

def cross_validate(session, train_x_all, train_y_all, cross_hidden_layers, cross_epochs, split_size=5):
  results = []
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_cross_x = train_x_all[train_idx]
    train_cross_y = train_y_all[train_idx]
    val_cross_x = train_x_all[val_idx]
    val_cross_y = train_y_all[val_idx]

    model = TrainModel(train_cross_x, train_cross_y, cross_hidden_layers, cross_epochs)
    cross_loss, cross_acc = model.evaluate(val_cross_x, val_cross_y)
    
    results.append(cross_acc)
  return results


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

    testSignificance = False

    crossValidate = False

    testOptimizationFrameworks = False

    #load mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    class_names = ['0', '1', '2', '3', '4', 
                '5', '6', '7', '8', '9']

    #plt.ion()
    #plt.figure()
    #plt.imshow(X_train[0])
    #plt.colorbar()
    #plt.grid(False)
    #plt.savefig('firstimageScale.png')
    
    #fig = plt.figure()
    #for i in range(16):
    #    plt.subplot(4,4,i+1)
    #    plt.tight_layout()
    #    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    #    plt.title("Digit: {}".format(y_train[i]))
    #    plt.xticks([])
    #    plt.yticks([])
    #fig
    #plt.savefig('exampleImages.png')

    #normalise data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    ##### TEST GAN

    model1 = TrainModel(X_train, y_train, 200, 30)
    test_loss, test_acc = model1.evaluate(X_test, y_test)
    print('Accuracy on test model:%.4f' % test_acc)

    lr = np.array(['0.0002beta']) #['0.1', '0.02', '0.002', '0.0002', '0.00002'])    
    ep = np.array(['1','30','50'])
    beta = np.array(['0.1','0.3','0.7','0.9'])
    bt = np.array(['4','16','64','256'])
    opt = np.array(['Adagrad','AdamSGD','GDO','RMSProp','SGD'])

    result = list()

    #for lr_counter in lr:
    #for lr_counter in beta:
    for lr_counter in lr:
        for ep_counter in ep:

            fName = '../GeneratedData_LearningRate_Variance/MNIST_cDCGAN_results_lr' + lr_counter + '/Epoch'+ ep_counter + '.npy'
            #fName = '../Generated_For_Classification_Batc/MNIST_cDCGAN_results_batch' + lr_counter + '/Epoch'+ ep_counter + '.npy'
            #fName = '../GeneratedDataForClassifier_Optimizer/MNIST_cDCGAN_results_' + lr_counter + '/Epoch'+ ep_counter + '.npy'

            #dataGAN = np.load('../GeneratedData_LearningRate_Variance/MNIST_cDCGAN_results_standard_lr_0.0002_momentum_0.5_batch_100_Adam/Epoch1.npy')
            dataGAN = np.load(fName)
   
            for i in range (10):
                X_gan = dataGAN[100 * i :  100 * i + 100 :]
                y_gan = np.empty(100)
                y_gan.fill(i)

                X_gan = X_gan + 1.0
                X_gan = X_gan /2.0
                #np.reshape(X_gan, -1)
                X_gan = np.reshape(np.ravel(X_gan), (100, 28, 28))

                #fig = plt.figure()
                #for i_plt in range(16):
                #    plt.subplot(4,4,i_plt+1)
                #    plt.tight_layout()
                #    plt.imshow(np.reshape(X_gan[i_plt], (28, 28)), cmap='gray')  #X_gan[i_plt], cmap='gray', interpolation='none')
                #    plt.title("Digit: {}".format(y_gan[i_plt]))
                #    plt.xticks([])
                #    plt.yticks([])
                #fig                
                #plt.savefig('./GANTest/GeneratedData_LearningRate_Variance/exampleImagesGAN_lr' + lr_counter + 'ep' + ep_counter + '_' + str(i) + '.png')
                ##plt.show()

                test_loss, test_acc = model1.evaluate(X_gan, y_gan)
                result.append(test_acc)

    result = np.asarray(result)
    result = np.reshape(result, (len(lr) * len(ep), 10))
    np.save('./GANTest/GeneratedData_LearningRate_Variance/test_result_beta.npy', result)
    np.savetxt("./GANTest/GeneratedData_LearningRate_Variance/test_result_beta.csv", result, delimiter=",")
    

    #saver = tf.train.Saver()     
    
    #exists = os.path.isfile(fileName)
    #if exists:
    #       os.remove(fileName)
    #save_path = saver.save(sess,'./models/mnist_2.ckpt') 

    if testMultipleParams:

        # grid search for a suitable parameter for the hidden layers and epochs
        number_of_hidden_layers = np.array([10, 20, 30, 40, 50, 100, 150, 200])
        number_of_epochs = np.array([1, 2, 3, 4, 5, 10, 20, 30])

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

        plotGridResults (number_of_epochs, number_of_hidden_layers, accuracies, 'GridSearch_3.png')

        plotGridResults (number_of_epochs, number_of_hidden_layers, timings, 'GridSearch_3_timings.png')

    if testOptimizationFrameworks:

                 
         losses1 = list()
         accuracies1 = list()
         timings1  = list()

         learning_rates = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])

         text_file = open("Output_optimizer.txt", "w")

         for learning_rate in learning_rates:    
             text_file.write('Learning rate %.3f \n' % learning_rate)     
             print('Learning rate %.5f ' % learning_rate)

             # Adam Optimizer
             text_file.write('Optimizer: Adam optimizer ')     
             print('Optimizer: Adam optimizer: ')
             time1 = time.time()
             optimizerAdam = tf.train.AdamOptimizer(learning_rate)    
             model = TrainModelOptimizer(X_train, y_train, 128, 10, optimizerAdam)
             time2 = time.time()
             time_span = time2 - time1
             timings1.append(time_span)
             print('Training the model took: %s seconds' % time_span)
             text_file.write('%.3f sec ' % time_span)         
                 
             # check accuracy
             test_loss, test_acc = model.evaluate(X_test, y_test)         
             losses1.append(test_acc)
             accuracies1.append(test_acc)
             print('Test accuracy: ', test_acc)
             text_file.write("%.3f\n" % test_acc)
                  
             # Grad descent Optimizer
             text_file.write('Optimizer: Gradient Descent ')    
             print('Optimizer: Gradient Descent ')    
             optimizerSGD = tf.train.GradientDescentOptimizer(learning_rate)    

             time1 = time.time()
             model = TrainModelOptimizer(X_train, y_train, 128, 10, optimizerSGD)
             time2 = time.time()
             time_span = time2 - time1
             timings1.append(time_span)
             print('Training the model took: %s seconds' % time_span)
             text_file.write('%.3f sec ' % time_span)         

             # check accuracy
             test_loss, test_acc = model.evaluate(X_test, y_test)         
             losses1.append(test_acc)
             accuracies1.append(test_acc)
             print('Test accuracy: ', test_acc)
             text_file.write("%.3f\n" % test_acc)

             # RMSPropOptimizer Optimizer
             text_file.write('Optimizer: RMSPropOptimizer ')               
             print('Optimizer: RMSPropOptimizer ')             
             optimizerRMS = tf.train.RMSPropOptimizer(learning_rate=learning_rate)    
    
             time1 = time.time()
             model = TrainModelOptimizer(X_train, y_train, 128, 10, optimizerRMS)
             time2 = time.time()
             time_span = time2 - time1
             timings1.append(time_span)
             print('Training the model took: %s seconds' % time_span)
             text_file.write('%.3f sec ' % time_span)       
         
             # check accuracy
             test_loss, test_acc = model.evaluate(X_test, y_test)         
             losses1.append(test_acc)
             accuracies1.append(test_acc)
             print('Test accuracy: ', test_acc)
             text_file.write("%.3f\n" % test_acc)

             # Momentum Optimizer
             text_file.write('Optimizer: Momentum ') 
             print('Optimizer: Momentum ')                 
             optimizerMomentum = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = 0.9)    
    
             time1 = time.time()
             model = TrainModelOptimizer(X_train, y_train, 128, 10, optimizerMomentum)
             time2 = time.time()
             time_span = time2 - time1
             timings1.append(time_span)
             print('Training the model took: %s seconds' % time_span)
             text_file.write('%.3f sec ' % time_span)       
         
             # check accuracy
             test_loss, test_acc = model.evaluate(X_test, y_test)         
             losses1.append(test_acc)
             accuracies1.append(test_acc)
             print('Test accuracy: ', test_acc)
             text_file.write("%.3f\n" % test_acc)

             # AdagradOptimizer
             text_file.write('Optimizer: Adagrad ')
             print('Optimizer: Adagrad ')      
             optimizerAdagrad = tf.train.AdagradOptimizer(learning_rate=learning_rate)    
    
             time1 = time.time()
             model = TrainModelOptimizer(X_train, y_train, 128, 10, optimizerAdagrad)
             time2 = time.time()
             time_span = time2 - time1
             timings1.append(time_span)
             print('Training the model took: %s seconds' % time_span)
             text_file.write('%.3f sec ' % time_span)       
         
             # check accuracy
             test_loss, test_acc = model.evaluate(X_test, y_test)         
             losses1.append(test_acc)
             accuracies1.append(test_acc)
             print('Test accuracy: ', test_acc)
             text_file.write("%.3f\n" % test_acc)

         text_file.close()

         #try plot                  
         fig = plt.figure()   
         x = ['Adam', 'Grad', 'RMS', 'Momentum', 'Adagrad']
         for i in range(0,6):
             c = [accuracies1[i * 5 + index] for index in range(0,5)]   # because there are 5 Optimizers
             label = learning_rates[i]
             plt.plot(x, c, label=label)
         plt.ylabel('Accuracy')
         plt.xlabel('Optimizer')
         plt.title('Accuracy depending on the learning rate in different Optimizers. Number of epochs - 10, Number of hidden layers - 128')
         plt.legend()
         fig.show()
         fig.savefig('Optimizers_learningrate.png')
         
    if testSignificance:
                  
        text_file = open("statistical_significance.txt", "w")

        for i in range (0, 10):
        
        #num_classes = 10
        #y_train = np_utils.to_categorical(y_train, num_classes)
        #y_test = np_utils.to_categorical(y_test, num_classes)

            model = TrainModel(X_train, y_train, 200, 30)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)    
            #model = TrainModelOptimizer(X_train, y_train, 128, 5, optimizer)

            # check accuracy
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print('Test accuracy:', test_acc)
            text_file.write("%.3f\n" % test_acc)

        text_file.close()
        #predictions = model.predict(X_test)    
        #likeliestPred = np.argmax(predictions[0])   #most likely prediction
        #print('First item is most likely:', np.argmax(predictions[0]))

    if cross_validate:
       
        folds = 5
        result = []

        hls = np.array([10, 20, 30, 40, 50, 100, 150, 200])
        #eps = np.array([5, 6, 7])

        for hl in hls:
            #for ep in eps:
                with tf.Session() as session:
                  result.append(cross_validate(session, X_train, y_train, hl, 10, split_size=folds))
        
        result = np.reshape(result, ( len(hls), folds))
        np.savetxt("crossvalidation_result_animals.csv", result, delimiter=",")

        mean = np.mean(result, axis = 1)
        std = np.std(result, axis = 1)
       
        fig, ax = plt.subplots()
        ax.bar(range(len(hls)), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(len(hls)))
        ax.set_xticklabels(hls)
        ax.set_title('Confidence intervals')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('bar_plot_with_error_bars.png')
        plt.show()   

        #print('Test accuracy: %f' % session.run(accuracy, feed_dict={x: test_x, y: test_y}))

