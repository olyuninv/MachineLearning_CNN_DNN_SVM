import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm

# Try comment

def plotData(digit_images,y):
  
  images_and_labels = list(zip(digit_images, y))
  fig, ax = plt.subplots(2, 4)
  for index, (image, label) in enumerate(images_and_labels[:8]):    
    ax[index//4,index%4].axis('off')
    ax[index//4,index%4].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax[index//4,index%4].set_title('Training: %i' % label)
  fig.savefig('digits.png')

def computeScore(X,y,preds):
  # for training data X,y it calculates the number of correct predictions made by the model
  ySize=y.shape[0]

  rest=y-preds
  score=ySize-np.count_nonzero(rest)
  
  #score=0
  return score

def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale, scale)
  
def main():
 
  n_samples=1797
  digit_images = np.loadtxt('digit_images.txt')
  digit_images = digit_images.reshape((n_samples, 8, 8))
  X = digit_images.reshape((n_samples, -1))
  y = np.loadtxt('digit_classes.txt')
  y_plot = y.reshape((n_samples, -1))
  
  # plot the first 8 images
  plotData (digit_images, y_plot)
  
  #create model svm
  #model = svm.SVC(C=0.5, gamma=0.75,kernel='polynomial')
  #model = svm.SVC(C=0.5, gamma=0.75,kernel='rbf')
  model = svm.SVC(C=0.5, gamma=0.75,kernel='sigmoid')

  # split the data into training and test parts - 10% test  
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.9)
  
  # now train the model ...
  model.fit(X_train,y_train)
  
  y_result = model.predict(X_test)

  target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
  print(classification_report(y_test, y_result, target_names=target_names))

  # add a column of ones to input data
  m=len(y)
  Xt = np.column_stack((np.ones((m, 1)), X))

  score = computeScore(Xt,y_test,y_result)
  print("score")
  print(score)

  from sklearn.model_selection import cross_val_score
  model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
  C_s, gamma_s = np.meshgrid(np.logspace(-2, 1, 20), np.logspace(-2, 1, 20))
  scores = list()
  i=0; j=0
  for C, gamma in zip(C_s.ravel(),gamma_s.ravel()):
    model.C = C
    model.gamma = gamma
    this_scores = cross_val_score(model, X_train, y_train, cv=5)
    scores.append(np.mean(this_scores))
  scores=np.array(scores)
  scores=scores.reshape(C_s.shape)
  fig2, ax2 = plt.subplots(figsize=(12,8))
  c=ax2.contourf(C_s,gamma_s,scores)
  ax2.set_xlabel('C')
  ax2.set_ylabel('gamma')
  fig2.colorbar(c)
  fig2.savefig('crossval.png')
  
if __name__ == '__main__':
  main()