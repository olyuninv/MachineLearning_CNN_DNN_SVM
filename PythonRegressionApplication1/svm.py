import numpy as np
#import sys
#sys.path.append("../Common/")
#from Data import DataReader
import matplotlib.pyplot as plt
#import mnist

from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm, metrics


# Try comment
#def plotData(digit_images,y):
  
#  images_and_labels = list(zip(digit_images, y))
#  fig, ax = plt.subplots(2, 4)
#  for index, (image, label) in enumerate(images_and_labels[:8]):    
#    ax[index // 4,index % 4].axis('off')
#    ax[index // 4,index % 4].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    ax[index // 4,index % 4].set_title('Training: %i' % label)
#  fig.savefig('digits.png')

#def computeScore(X,y,preds):
#  # for training data X,y it calculates the number of correct predictions made
#  # by the model
#  ySize = y.shape[0]

#  rest = y - preds
#  score = ySize - np.count_nonzero(rest)
  
#  #score=0
#  return score

#def normaliseData(x):
#  # rescale data to lie between 0 and 1
#  scale = x.max(axis=0)
#  return (x / scale, scale)

  
def main():

  train_test = False
  train_RBF = True
  train_linear = False
  train_polynomial = False
  train_rbf_2 =True

  #n_samples = 1797
  #digit_images = np.loadtxt('digit_images.txt')
  #digit_images = digit_images.reshape((n_samples, 8, 8))
  #X = digit_images.reshape((n_samples, -1))
  #y = np.loadtxt('digit_classes.txt')
  #y_plot = y.reshape((n_samples, -1))
  
  # plot the first 8 images
  #plotData(digit_images, y_plot)

  #dataLion = DataReader.readImages('../../Data/n02118333/', 0, 28, True)
  #dataFox = DataReader.readImages('../../Data/n02129165/', 1, 28, True)
  #dataTortoise = DataReader.readImages('../../Data/n01669191/', 2, 28, True)
  #data0 = DataReader.readImages('../../Data2/0/', 0, 28, True)
  #data1 = DataReader.readImages('../../Data2/1/', 1, 28, True)
  #data2 = DataReader.readImages('../../Data2/2/', 2, 28, True)
  #data3 = DataReader.readImages('../../Data2/3/', 0, 28, True)
  #data4 = DataReader.readImages('../../Data2/4/', 1, 28, True)
  #data5 = DataReader.readImages('../../Data2/5/', 2, 28, True)
  #data6 = DataReader.readImages('../../Data2/6/', 0, 28, True)
  #data7 = DataReader.readImages('../../Data2/7/', 1, 28, True)
  #data8 = DataReader.readImages('../../Data2/8/', 2, 28, True)
  #data9 = DataReader.readImages('../../Data2/9/', 2, 28, True)
  
  # Keep first 1200
  #dataLion = dataLion[:1200]
  #dataFox = dataFox[:1200]
  #dataTortoise = dataTortoise[:1200]
  #data0 = data0[:1200]
  #data1 = data1[:1200]
  #data2 = data2[:1200]
  #data3 = data3[:1200]
  #data4 = data4[:1200]
  #data5 = data5[:1200]
  #data6 = data6[:1200]
  #data7 = data7[:1200]
  #data8 = data8[:1200]
  #data9 = data9[:1200]

  #dataset = np.concatenate([dataLion, dataFox, dataTortoise])
  #dataset = np.concatenate([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9])
  #np.random.shuffle(dataset)

  #train_x = list()
  #train_y = list()
    
  # split into x and y
  #for (X, y) in dataset:
  #    train_x.append(X)
  #    train_y.append(y)

  #train_x = np.asarray(train_x)
  #train_y = np.asarray(train_y)
  #print("Successfully loaded data")
  
  # split the data into training and test parts - 10% test
  ##X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
  #X_train, X_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.1)

  #X_train=X_train.reshape(10800,784)
  #X_test=X_test.reshape((1200),784)
  #X_train=X_train.reshape(3240,16384)
  #X_test=X_test.reshape((3600-3240),16384)
  #if train_rbf_2:

  digits = datasets.load_digits()
  images_and_labels = list(zip(digits.images, digits.target))

  import matplotlib.pyplot as plt
  for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

  n_samples = len(digits.images)
  data = digits.images.reshape((n_samples, -1))

  X = data
  y = digits.target

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  model = svm.SVC(gamma=0.001)
  #learn digits
  model.fit(X_train,y_train)
  #predict value of digits
  expected = y_test
  predicted = model.predict(X_test)

  print("Classification report for classifier %s:\n%s\n"
        % (model, metrics.classification_report(expected, predicted)))
  print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
  print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

  images_and_predictions = list(zip(digits.images[X_train.shape[0]:], predicted))
  for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

  plt.show()
      
  if train_test:
      #create model svm
      #model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      model = svm.SVC(C=1.0, cache_size=1000, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

      # now train the model ...
      model.fit(X_train,y_train)
  
      y_result = model.predict(X_test)

      target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
      #target_names = ['lion','fox','turtle']
      print(classification_report(y_test, y_result, target_names=target_names))
      

  if train_RBF:
      import matplotlib.pyplot as plt

      C_s, gamma_s = np.meshgrid(np.logspace(-2, -1, 20), np.logspace(-2, 1, 20))
      #C_s, gamma_s = np.meshgrid(np.logspace(-2, -0, 10), np.logspace(-5,-2, 10))
      scores = list()
      i = 0
      j = 0
      for C_param, gamma_param in zip(C_s.ravel(),gamma_s.ravel()):
        model.C = C_param
        model.gamma = gamma_param

        model = svm.SVC(C=C_param, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=gamma_param, kernel='rbf',
            max_iter=-1, probability=False)
        this_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1)
        scores.append(np.mean(this_scores))
      
      out_file = open('Scores_rbf', 'w')
      for i in range(0, 10):
          out_file.write('\n%.5f,\n' % gamma_s[i, 0])  
          for j in range(0, 10):
              out_file.write('%.5f,' % C_s[0, j])
              out_file.write('%.3f,' % scores[i*10 + j])
             
              
              
      out_file.close()
                    
      scores = np.array(scores)
      scores = scores.reshape(C_s.shape)
      fig2, ax2 = plt.subplots(figsize=(12,8))
      c = ax2.contourf(C_s,gamma_s,scores)
      ax2.set_xlabel('C')
      ax2.set_ylabel('gamma')
      fig2.colorbar(c)
      fig2.show()
      fig2.savefig('crossval.png')
  
  if train_linear:

      model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

      # now train the model ...
      model.fit(X_train,y_train)
  
      y_result = model.predict(X_test)

      target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
      print(classification_report(y_test, y_result, target_names=target_names))


      C_s = np.logspace(-8, -2, 10)

      scores = list()
      scores_std = list()
      for C_param in C_s:
          model = svm.SVC(C=C_param,kernel='linear')
          this_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1)
          scores.append(np.mean(this_scores))
          scores_std.append(np.std(this_scores))

      ## Do the plotting
      import matplotlib.pyplot as plt

      plt.figure(1, figsize=(4, 3))
      plt.clf()
      plt.semilogx(C_s, scores)
      locs, labels = plt.yticks()
      plt.ylabel('CV score')
      plt.xlabel('Parameter C')
      plt.ylim(0, 1.1)
      plt.show()

  if train_polynomial:
      import matplotlib.pyplot as plt

      C_s, gamma_s = np.meshgrid(np.logspace(-2, -0, 10), np.logspace(-5,-3.5, 10))
      scores = list()
      i = 0
      j = 0
      for C_param, gamma_param in zip(C_s.ravel(),gamma_s.ravel()):
        model.C = C_param
        model.gamma = gamma_param

        model = svm.SVC(C=C_param, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=gamma_param, kernel='poly',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        this_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1)
        scores.append(np.mean(this_scores))
      
      scores = np.array(scores)
      scores = scores.reshape(C_s.shape)
      
      fig3, ax3 = plt.subplots(figsize=(12,8))
      c = ax3.contourf(C_s,gamma_s,scores)
      ax3.set_xlabel('C')
      ax3.set_ylabel('gamma')
      fig3.colorbar(c)
      fig3.show()
      fig3.savefig('crossval.png')


if __name__ == '__main__':
  main()