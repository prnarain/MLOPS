
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

def digitsClassifier(data,gamma=0.001):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    # split data into 70% train and 30% (test + val) subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    
    # split test into test(15%) and val(15%)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    accuracy_on_test= round(accuracy_score(y_test, predicted), 4)  
    f1_on_test = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)

    # Predict the value of the digit on the train subset
    predicted = clf.predict(X_train)
    accuracy_on_train= round(accuracy_score(y_train, predicted), 4)  
    f1_on_train = round(f1_score(y_train, predicted, average='macro', zero_division=0), 4)

    # Predict the value of the digit on the val subset
    predicted = clf.predict(X_val)
    accuracy_on_val= round(accuracy_score(y_val, predicted), 4)  
    f1_on_val = round(f1_score(y_val, predicted, average='macro', zero_division=0), 4)
    
    return [[accuracy_on_train,f1_on_train],[accuracy_on_test,f1_on_test],[accuracy_on_val,f1_on_val]]
    
# checking for different gamma values

data_org = digits.images
best_gamma=0
max_accuracy=0
for gamma in [0.5,0.01,1,0.001,0.0001,0.000005]:
    ans = []
    print(f"for gamma = {gamma}")
    ans.append(digitsClassifier(data_org, gamma=gamma))
    for a in ans:
        print("Accuracy score on train set ",end=" ")
        print(a[0][0])
        print("Accuracy score on test set ",end=" ")
        print(a[1][0])
        print("Accuracy score on val set ",end=" ")
        print(a[2][0])
        print()

        if a[2][0]>max_accuracy:
          best_gamma=gamma
          max_accuracy=a[2][0]

print("Maximum accuracy is found at gamma value= ",best_gamma)