
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from skimage.transform import rescale

import os
import numpy as np
from joblib import dump, load

## functions - preprocess - create split - test 

def preprocess(images,rescale_factor):
  resized_images=[]
  for image in images:
    resized_images.append(rescale(image, rescale_factor, anti_aliasing=False))
  return resized_images

## train test split 
def createsplit(data,targets,test_size,valid_size):
  X_train, X_test_valid, y_train, y_test_valid = train_test_split(data, targets, test_size=test_size + valid_size, shuffle=False)
  X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,y_test_valid,test_size=valid_size / (test_size + valid_size),shuffle=False)
  return X_train, X_test,X_valid,y_train,y_test,y_valid

## predict test 
def test(clf,X,y):
  predicted = clf.predict(X)
  acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
  f1 = metrics.f1_score(y_pred=predicted, y_true=y, average="macro")
  return {'accuracy':acc,'f1score':f1}