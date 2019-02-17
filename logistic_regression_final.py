# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 02:18:37 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 01:07:25 2019

@author: user
"""

import numpy as np
import pandas as pd
import naive_bayes_final as nb
from sklearn.model_selection import train_test_split

#alpha is the constant parameter Ao we are assuming Ao + sigma(Bi), where i ranges from 1 to m.

def predict(features, beta, alpha):
    weights = np.dot(beta, features.T) + alpha     ##get the updated value for the weights
  #  print("weights shape = " , str(weights.shape))
    predicted_labels = 1/(1+np.exp(-weights))     ##get the predicted labels using the updated weights
    return predicted_labels
    
'''
theta : parameter to control gradient descent. 
diff: the difference in weight vector. we will stop when the magnitude of the difference between 
current estimates of beta vs revised estimates become <= lim. 
'''
def update_weights(theta, features, actual_labels, beta, alpha):
    
    m = len(features)
    print("Training...")
    e = 1
    count = 0
  #  num_iter = 1000000
    while e >= 0.000001:                                 ##while error >= limit.
      predicted_labels = predict(features, beta, alpha)
      prev_beta = beta
     # print("predicted labels = " + str(predicted_labels))
      diff = actual_labels-predicted_labels     ##diff is 1x9 array, one predicted label for each data point.
      grad = np.dot(diff, features)/m         ##features is 9x4. So grad = 1x9 x 9x4 = 1x4
     # print("grad = " , str(grad))
     # print("Iteration = " , str(count) , " Error = " , str(e))
      beta = beta + theta*grad
      e =  np.sqrt(np.sum((beta-prev_beta)**2))
      count += 1
    print("Done! Num of iterations = " , str(count))
    return beta
'''
100 iterations : [0.62726654 0.71930482 0.05080914 0.07185464]
200 iterations : [0.69691348 0.82711054 0.0307152  0.0540531 ]
300 iterations : [0.74212875 0.88227212 0.02128699 0.04292157]
400 
'''

def train_and_test(features_train, test_data):    
    
    features = features_train[:,:-1]
    actual_labels = features_train[:,-1]
    n = len(features)
#    print("Length of train set : " , len(features))
    beta = np.array([0,0,0,0])                    ##1x4. Features is 9x4
    theta = 0.000001                          ##beta*feature.T is 1x9
    alpha = np.array([0.001]*n)
    
    mis_classify_train = 0
    beta = update_weights(theta, features, actual_labels, beta, alpha)
    predicted_labels = predict(features, beta, alpha)
    
    for i in range(0,len(predicted_labels)):
        if predicted_labels[i] >= 0.5:
            predicted_labels[i] = 1
        else:
            predicted_labels[i] = 0
        if(actual_labels[i] != predicted_labels[i]):
            mis_classify_train += 1
    
    total_train = len(features)
    accuracy_train = (1-(mis_classify_train/total_train))*100
    print("For Logistic Regression: ")
    print("Accuracy of the system on train set : " + str(accuracy_train))
    
    ##NOW OBTAIN ACCURACY ON TEST SET
    features_test = test_data[:,:-1]        ##get labels into one numpy 2d array
    actual_labels_test = test_data[:,-1]
#    print("Length of test set : " , len(features_test))
    n = len(features_test)
    alpha_test = np.array([0.001]*n)
   # print(alpha_test[:4])
    mis_classify_test = 0
    
    predicted_labels_test = predict(features_test, beta, alpha_test)
    for i in range(0,len(predicted_labels_test)):
        if predicted_labels_test[i] >= 0.5:
            predicted_labels_test[i] = 1
        else:
            predicted_labels_test[i] = 0
        if(actual_labels_test[i] != predicted_labels_test[i]):
            mis_classify_test += 1
            
    total_test = len(features_test)
    accuracy_test = (1-(mis_classify_test/total_test))*100
    print("Accuracy of the system on test set : " + str(accuracy_test))
    return accuracy_test

    
'''
theta = 0.000000001
 lim = 0.00000001
 prev_beta = beta                             ##store the current values of parameter estimates

   SECOND WAY OF PICKING THE RANDOM SAMPLES AND TESTING THE ALGORITHM.
   features = data[train][:,:-1]
   labels = data[train][:,-1]
   train_x,test_x,train_y,test_y = train_test_split(features,labels,test_size=1-f,random_state=None)
   test_x = data[test][:,:-1]
   test_y = data[test][:,-1]
   accuracy_rate = train_and_test(train_x,train_y,test_x,test_y)
'''