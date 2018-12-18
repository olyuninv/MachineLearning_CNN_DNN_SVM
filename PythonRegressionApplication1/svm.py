import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm, metrics

def main():

  train_test = False
  train_RBF = False
  train_linear = False
  train_polynomial = True

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
      C_s, gamma_s = np.meshgrid(np.logspace(-2.5, 0, 15), np.logspace(-5,-2, 15))
      scores = list()
      i = 0
      j = 0
      data_barchart = []

      for C_param, gamma_param in zip(C_s.ravel(),gamma_s.ravel()):
        model.C = C_param
        model.gamma = gamma_param

        model = svm.SVC(C=C_param, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=gamma_param, kernel='rbf',
            max_iter=-1, probability=False)
        this_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1)
        data_barchart.append(this_scores)
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

      #confidence interval
      data_barchart = np.reshape(data_barchart, ( 15*15, 5))
      #data_barchart=data_barchart[:10]
      data_barchart=data_barchart[50:60]
      np.savetxt("crossvalidation_result_mnist.csv", scores, delimiter=",")

      mean = np.mean(data_barchart, axis = 1)
      std = np.std(data_barchart, axis = 1)
       
      fig, ax = plt.subplots()
      ax.bar(range(10), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
      ax.set_ylabel('Accuracy')
      ax.set_xticks(range(len(data_barchart)))
      ax.set_xticklabels(['a','b','c','c','c','c','c','c','c','c'])
      ax.set_title('C')
      ax.yaxis.grid(True)
      # Save the figure and show
      plt.tight_layout()
      plt.savefig('bar_error_mnist_rbf.png')
      plt.show()   

      fig2, ax2 = plt.subplots(figsize=(12,8))
      c = ax2.contourf(C_s,gamma_s,scores,np.arange(0.0, 1.1, .1))
      ax2.set_xlabel('C')
      ax2.set_ylabel('gamma')
      bounds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      fig2.colorbar(c, boundaries=bounds, ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
      fig2.show()
      fig2.savefig('RBF.png')
  
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

      #C_s, gamma_s = np.meshgrid(np.logspace(-5, -1, 10), np.logspace(-3, -1, 10))
      #C_s, gamma_s = np.meshgrid(np.logspace(-4, 0, 10), np.logspace(-4.2, -3.7, 10))
      C_s, gamma_s = np.meshgrid(np.logspace(-6, 2, 10), np.logspace(-6, -1, 10))
      scores = list()
      i = 0
      j = 0

      data_barchart = []

      for C_param, gamma_param in zip(C_s.ravel(),gamma_s.ravel()):
        model.C = C_param
        model.gamma = gamma_param

        model = svm.SVC(C=C_param, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=gamma_param, kernel='poly',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        this_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=1)
        data_barchart.append(this_scores)
        scores.append(np.mean(this_scores))
      
      out_file = open('Scores_poly', 'w')
      for i in range(0, 10):
          out_file.write('\n%.5f,\n' % gamma_s[i, 0])  
          for j in range(0, 10):
              out_file.write('%.5f,' % C_s[0, j])
              out_file.write('%.3f,' % scores[i*10 + j])
      out_file.close()
          
      scores = np.array(scores)
      scores = scores.reshape(C_s.shape)

      #confidence interval
      data_barchart = np.reshape(data_barchart, ( 10*10, 5))
      #data_barchart=data_barchart[:10]
      data_barchart=data_barchart[50:60]
      np.savetxt("crossvalidation_result_mnist.csv", scores, delimiter=",")

      mean = np.mean(data_barchart, axis = 1)
      std = np.std(data_barchart, axis = 1)
       
      fig, ax = plt.subplots()
      ax.bar(range(10), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
      ax.set_ylabel('Accuracy')
      ax.set_xticks(range(len(data_barchart)))
      ax.set_xticklabels(['a','b','c','c','c','c','c','c','c','c'])
      ax.set_title('C')
      ax.yaxis.grid(True)
      # Save the figure and show
      plt.tight_layout()
      plt.savefig('bar_error_mnist_poly.png')
      plt.show()   

      #gamma graph

      mean = scores.mean(axis=1)
      std = scores.std(axis=1)
       
      fig, ax = plt.subplots()
      ax.bar(range(10), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
      ax.set_ylabel('Accuracy')
      ax.set_xticks(range(10))
      ax.set_xticklabels(['a','b','c','c','c','c','c','c','c','c'])
      ax.set_title('C')
      ax.yaxis.grid(True)

      # Save the figure and show
      plt.tight_layout()
      plt.savefig('bar_error_mnist_poly.png')
      plt.show()   
      
      fig3, ax3 = plt.subplots(figsize=(12,8))
      #c = ax3.contourf(C_s,gamma_s,scores,np.arange(0.95, 1.0, .005))
      c = ax3.contourf(C_s,gamma_s,scores,np.arange(0.0, 1.05, .05))
      ax3.set_xlabel('C')
      ax3.set_ylabel('gamma')
      bounds=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      fig3.colorbar(c, boundaries=bounds, ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
      fig3.show()
      fig3.savefig('POLY.png')


if __name__ == '__main__':
  main()