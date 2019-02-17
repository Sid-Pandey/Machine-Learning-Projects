# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:21:00 2019

@author: user
"""

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
from sklearn.model_selection import KFold

'''
This function receives a 1x4 feature vector corresponding to a single data point. 
It calculates the Gaussian Naive bayes function to estimate the likelihood P(X | Y = 1) or P(X | Y = 0), 
depending on parameters means, sd and prior. 

Parameters : 
data_array : A 1x4 feature vector for a single data point.
means : A 1x4 vector containing means for the 4 individual features, for one class {0 or 1}
sd : A 1x4 vector containing standard deviations for the 4 individual features, for one class {0 or 1}

Output:
ans = The log likelihood estimate of the data point belonging to class c {0 or 1}
'''
def predict(data_array, means, sd, prior):
    prob = 1;
    for i in range(0,len(data_array)):
        x = data_array[i]
        prob = prob * 1/(sd[i]*np.sqrt(2*np.pi)) * ( np.exp( -1*np.square(x-means[i]) / (2*np.square(sd[i])) ) )
    
#    ans = np.log(prior) + np.log(prob)
    ans = prior*prob
    return ans

'''
This function takes as input the feature vector consisting of all data points in the test set. 
It returns the likelihood of a data point X belonging to a particular class {0,1}
This function is called 2 times for a given data point X, once for each class 
Parameters : 
data : The input feature vector
means : A 1x4 vector containing means for the 4 individual features, for one class {0 or 1}
sd : A 1x4 vector containing standard deviations for the 4 individual features, for one class {0 or 1}
prior : The prior probability of the class {0 or 1}

Output:
predictions : An array containing the probability of all the data points belonging to a particular class {0 or 1}
depending on the input parameters {means, sd, prior}

This function uses predict() function as a subroutine. 
'''
def get_predictions(data, means, sd, prior):
#    print("Features[0] has length : " , len(data[0]))
    predictions = np.apply_along_axis(predict, 1, data, means, sd, prior)
    return predictions

'''
This function receives the train and test data and does the following:
    
1. Receives the train data and splits it into 2 classes, data points with label 1 and those with label 0

2. For each set of data points, calculates the mean and standard deviation for all the features. Stores 
these values in the arrays 
means_0 : 1 x 4 : stores mean for all the 4 features, for data points belonging to class 0.
means_1 : 1 x 4 : stores mean for all the 4 features, for data points belonging to class 1.
sd_0 : 1 x 4 : stores standard deviations for all the 4 features, for data points belonging to class 0.
sd_1 : 1 x 4 : stores standard deviations for all the 4 features, for data points belonging to class 1.

3. It then calls the function get_predictions, first with parameters means_0, sd_0, prior_0 which returns 
the probability for all data points to belong to class 0. Store the result in array test_predictions_0.

4. Then the fuction get_predictions is called with parameters means_1, sd_1, prior_1 which returns 
the probability for all data points to belong to class 1. Stores the results in array test_predictions_1.

5. Then , for a data point i, if test_predictions_0[i] > test_predictions_1[i], it is assigned class 0, 
else it is assigned class 1.
'''
def train_and_test(features, features_test):    

    total = features.size
    original_classes = features[:,-1]       ##original values for class labels.
    original_features = features[:,:-1]     ##original values for the features. Needed for testing. 
    
    class_1 = features[:,4] == 1
    class_0 = features[:,4] == 0
    features_0 = features[class_0]
    features_1 = features[class_1]
#    print(type(features_0))

    prior_0 = features_0.size/total
    prior_1 = features_1.size/total
#    print(prior_0)
#    print(prior_1)
    

    num_features = len(features[0])-1           ##the last label contains the class. 
    means_1 = np.array(0*4)   ##arrays to store mean and standard deviation for the 4 features. 
    means_0 = np.array(0*4)
    sd_0 = np.array(0*4)
    sd_1 = np.array(0*4)
    for i in range(0,num_features):
        means_0 = np.mean(features_0,axis=0)
        means_1 = np.mean(features_1, axis=0)
        sd_0 = np.std(features_0,axis=0)
        sd_1 = np.std(features_1,axis=0)
 
    ##Finding the accuracy of the model on the test set. 
    test_features = features_test[:,:-1]
    test_classes = features_test[:,-1]
    total_test = len(test_features)
    test_predictions_0 = get_predictions(test_features, means_0, sd_0, prior_0)   ##take the entire training set. 
    test_predictions_1 = get_predictions(test_features, means_1, sd_1, prior_1)   ##calculate values for predictions 0 and 1.
    mis_classify_test = 0
    
    for i in range(0,len(test_predictions_0)):
        predicted = 0
        if test_predictions_1[i] > test_predictions_0[i]:
            predicted = 1
        if(predicted != test_classes[i]):
            mis_classify_test += 1
            #print("predictions[0]=" , test_predictions_0[i], " predictions[1]=" , test_predictions_1[i], " actual=" , test_classes[i])

    accuracy_rate = (1-mis_classify_test/total_test)*100    
    print("---GAUSSIAN NAIVE BAYES : ACCURACY RATE ON TEST SET---\n")
    print(accuracy_rate)
    return accuracy_rate, means_1, sd_1
       
#    func_test = 1/(2*np.sqrt(2*np.pi)) * ( np.exp( -1*np.square(4-2) / (2*np.square(2)) ) )
#    print(func_test)