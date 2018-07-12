# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 2018

@author: TheRealMentor
"""

#Importing package for euclidean distance
from scipy.spatial import distance

#Defining the function
def euc(a, b):
    return distance.euclidean(a,b)

#Defining the Scrappy Classifier
class ScrapyKNN():
    
    #Defining the fit method
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    #Defining the predict method
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    #Defining the closest method for finding the closest neighbour
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        
        return self.y_train[best_index]


#Importing the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

#Taking features(data) as X and labels(target) as y
X = iris.data
y = iris.target

#Splitting into the training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Defining the classifier
clf = ScrapyKNN()

#Training the classifier
clf.fit(X_train, y_train)

#Predicting using the test set
y_pred = clf.predict(X_test)

#Checking the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))