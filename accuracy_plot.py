# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:15:11 2019

@author: user
"""
import naive_bayes_final as nb
import logistic_regression_final as lr
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

'''
Function to generate 400 random samples for each of the 4 features and return their mean and variance. 
Mean and variance are returned as 1 x 4 vectors.
'''
def get_mean_and_variance(means_1, sd_1):
    
    new_means = []
    new_stds = []
    
    for i in range(0,len(means_1)):              ##going through mean and standard deviation for each feature
        data1 = np.random.normal(means_1[i], sd_1[i], 400)
        new_means.append(np.mean(data1))
        new_stds.append(np.std(data1))
    
    new_means = np.array(new_means)
    new_stds = np.array(new_stds)
    return new_means, new_stds

'''
This function does the following: 
1. Performs a three fold cross validation. Thus it divides the data into testing and training sets 3 times. 
2. For each iteration, it takes f random fraction of the training data, where f comes from the list [0.01,0.02,0.05,0.1,0.625,1]
3. For each of the fractions above, it runs the naive bayes and logistic regression algorithms 5 times and 
collects the accuracies from both in arrays 
accuracy_gnb : Gaussian Naive Bayes
accuracy_lr : Logistic regression
The arrays are multidimensional with dimensionality 3 x 6. The3 arrays are used to store the accuracy values
for each of the passes in k-fold cross validation.
'''
def random_split_testing(data, fraction):
        data = data.values
        kfold = KFold(3, True, 1) # enumerate splits
        counter = 0

        accuracy_gnb = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]]
        accuracy_lr = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        
        for train, test in kfold.split(data):       ##split up into training and test sets
            compare = True
            
            for i in range(0,6):                    ##consider each fraction for splitting up training data
                f = fraction[i]
                for j in range(0,5):
                    sampled_df = pd.DataFrame(data[train]).sample(frac=f, replace=False, random_state=None)
                    features = sampled_df.values
                    print("No of data points taken : " + str(len(features)))
                    features_test = data[test]
                    accuracy_rate = lr.train_and_test(features,features_test)
                    accuracy_lr[counter][i] += accuracy_rate
                    accuracy_rate, means_1, sd_1 = nb.train_and_test(features,features_test)
                    accuracy_gnb[counter][i] += accuracy_rate
                    
                    if(compare and f == 1):       ##compare mean and variance when using all train data.  
                        print("Original mean and variance : " + str(means_1) + "\n" + str(np.square(sd_1)))
                        means_1_new, sd_1_new = get_mean_and_variance(means_1, sd_1)
                        print("Mean and variance for randomly generated data : " + str(means_1_new) + "\n" + str(np.square(sd_1_new)))
                        compare = False
                        
            print("For " , str(counter) , " pass of cross validation the accuracies are : \n")
            for i in range(0,len(accuracy_lr[counter])):                     ##looking at one set of train,test data
                accuracy_lr[counter][i] /= 5
                accuracy_gnb[counter][i] /= 5
            print("For logistic regression : " , str(accuracy_lr))
            print("For gaussian naive bayes : " , str(accuracy_gnb))
            counter += 1
        print("Done! Program exiting..")
        return accuracy_gnb, accuracy_lr
    
'''
accuracy_lr and accuracy_gnb are 3 x 6 dimensional arrays having the accuracy for each of the fractions, 
for all the 3 folds. 
In this function, we compute the average for each fraction over the three folds, and plot curves for logistic 
regression and naive bayes. 
'''
def plot(fraction, accuracy_lr, accuracy_gnb):
    gnb_list = []  
    lr_list = []
    
    for j in range(0,6):
        avg_gnb = 0
        avg_lr = 0
        for i in range(0,len(accuracy_lr)):
            avg_gnb += accuracy_gnb[i][j]
            avg_lr += accuracy_lr[i][j]
        avg_gnb /= 3
        avg_lr /= 3
        gnb_list.append(avg_gnb)
        lr_list.append(avg_lr)
        
    print(gnb_list)
    print(lr_list)
    
    plt.plot(fraction, gnb_list, color='b', label = 'Gaussian Naive Bayes', marker='o')
    plt.plot(fraction, lr_list, color='g', label = 'Logistic Regression', marker='s')
    plt.xlabel('Fraction of training dataset used')
    plt.ylabel('Accuracy')
    plt.title('Learning curve : Accuracy vs Size of training set')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    data = pd.read_csv("banknote_dataset.txt")
    fraction = [0.01,0.02,0.05,0.1,0.625,1]
    accuracy_gnb, accuracy_lr = random_split_testing(data, fraction)
    plot(fraction, accuracy_lr, accuracy_gnb)