import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm, metrics

import sys
sys.path.append("./utils/")
sys.path.append("./data/")

from utils import mnist_reader

def main():

  train_test = False
  train_RBF = True
  train_linear = False
  train_polynomial = False

  X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
  X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

  X_train = X_train[:1200]
  y_train = y_train[:1200]
  X_test = X_test[:1200]
  y_test = y_test[:1200]

  #X_train = X_train[:10000]
  #y_train = y_train[:10000]
  #X_test = X_test[:10000]
  #y_test = y_test[:10000]

  X_train=X_train/255.0
  X_test=X_test/255.0


  #import matplotlib.pyplot as plt

  #plt.figure()
  #plt.imshow(np.reshape(X_train[0], (28, 28)))
  #plt.colorbar()
  #plt.grid(False)
  #plt.savefig('firstimageFashion.png')
    
  #fig = plt.figure()
  #for i in range(16):
  #      plt.subplot(4,4,i+1)
  #      plt.tight_layout()
  #      plt.imshow(np.reshape(X_train[i], (28, 28)), cmap='gray', interpolation='none')
  #      plt.title("Label: {}".format(y_train[i]))
  #      plt.xticks([])
  #      plt.yticks([])
  #fig
  #plt.savefig('exampleFashionImages.png')
  #plt.show()

  #model = svm.SVC(gamma=0.001)
  #model = svm.SVC(gamma=0.001)
  #model =svm.SVC(C=10, gamma=0.001) #accuracy 0.81
  model =svm.SVC(C=10)  #accuracy 0.81
  #model = svm.SVC(C=1,gamma=0.1, kernel='poly') #0.7683 accuracy
  #model = svm.SVC(gamma=0.0001, kernel='poly') #0.0975 accuracy
  #learn
  model.fit(X_train,y_train)
  #predict 
  expected = y_test
  predicted = model.predict(X_test)

  print("Classification report for classifier %s:\n%s\n"
        % (model, metrics.classification_report(expected, predicted)))
  conf_mat = metrics.confusion_matrix(expected, predicted)
  print("Confusion matrix:\n%s" % conf_mat)
  #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
  print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

  # Plot Confusion Matrix Data as a Matrix
  import matplotlib.pyplot as plt
  conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
  plt.figure(1, figsize=(7, 3), dpi=160)
  plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Confusion matrix')
  plt.colorbar()
  fmt = '.2f'
  thresh = conf_mat.max() / 2.
  for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')  
  plt.savefig("fashion_conf_matrix.png")
  plt.show()
      
  if train_test:
      #create model svm
      model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
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

      #C_s, gamma_s = np.meshgrid(np.logspace(-2.5, -0.5, 15), np.logspace(-5,-2, 15))
      #C_s, gamma_s = np.meshgrid(np.logspace(-0.5, 1.5, 15), np.logspace(-5,-2, 15))
      #C_s, gamma_s = np.meshgrid(np.logspace(0, 3, 10), np.logspace(-5,-2, 10))
      #C_s, gamma_s = np.meshgrid(np.logspace(-2, 3, 10), np.logspace(-7,-2, 10))
      C_s, gamma_s = np.meshgrid(np.logspace(-4, 1, 10), np.logspace(-9,-4, 10))

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
      
      out_file = open('Scores_rbf_fashion', 'w')
      for i in range(0, 3):
          out_file.write('\n%.5f,\n' % gamma_s[i, 0])  
          for j in range(0, 3):
              out_file.write('%.5f,' % C_s[0, j])
              out_file.write('%.3f,' % scores[i*3 + j])
      out_file.close()
                    
      scores = np.array(scores)
      scores = scores.reshape(C_s.shape)

      fig2, ax2 = plt.subplots(figsize=(12,8))
      c = ax2.contourf(C_s,gamma_s,scores,np.arange(0.0, 1.05, .05))
      ax2.set_xlabel('C')
      ax2.set_ylabel('gamma')
      bounds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      fig2.colorbar(c, boundaries=bounds, ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
      fig2.show()
      fig2.savefig('RBF_fashion.png')
  
  if train_linear:

      model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

      # now train the model ...
      model.fit(X_train,y_train)
  
      y_result = model.predict(X_test)

      target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
      print(metrics.classification_report(y_test, y_result, target_names=target_names))

      print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


      #C_s = np.logspace(-8, -2, 10)
      C_s = np.logspace(-8, 0, 10)

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
      plt.savefig('fashion_linear.png')
      plt.show()

  if train_polynomial:
      import matplotlib.pyplot as plt

      #C_s, gamma_s = np.meshgrid(np.logspace(-5, -1, 10), np.logspace(-3, -1, 10))
      #C_s, gamma_s = np.meshgrid(np.logspace(-3, 0, 3), np.logspace(-4, -3.7, 3))
      #C_s, gamma_s = np.meshgrid(np.logspace(-9, 7, 3), np.logspace(-7, 0, 3))
      #C_s, gamma_s = np.meshgrid(np.logspace(-12, 7, 3), np.logspace(-10, 0, 3))
      C_s, gamma_s = np.meshgrid(np.logspace(-15, 10, 3), np.logspace(-13, 2, 3))
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
      

      out_file = open('Scores_poly_fashion', 'w')
      for i in range(0, 3):
          out_file.write('\n%.5f,\n' % gamma_s[i, 0])  
          for j in range(0, 3):
              out_file.write('%.5f,' % C_s[0, j])
              out_file.write('%.3f,' % scores[i*3 + j])
      out_file.close()

      scores = np.array(scores)
      scores = scores.reshape(C_s.shape)
      
      fig3, ax3 = plt.subplots(figsize=(12,8))
      #c = ax3.contourf(C_s,gamma_s,scores,np.arange(0.95, 1.0, .005))
      c = ax3.contourf(C_s,gamma_s,scores,np.arange(0.0, 1.025, .025))
      ax3.set_xlabel('C')
      ax3.set_ylabel('gamma')
      bounds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      #norm = colors.BoundaryNorm(bounds, cmap.N)
      fig3.colorbar(c)#, boundaries=bounds,ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
      fig3.show()
      fig3.savefig('POLY_fashion.png')


if __name__ == '__main__':
  main()
