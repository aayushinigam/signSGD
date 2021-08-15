"""
FashionMNIST_logisticRegression - signSGD on Neural Nets
dataset - Fasion MNIST dataset

"""

# Importing libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle    
import os
import functools
import time
from sklearn.metrics import log_loss

# Loading data
(X_train, label_train), (X_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

# utility function to create one-hot encoding from labels and normalizing the data
def process_data(X, label):
    m = X.shape[0]
    assert(m == label.shape[0])
    X = X.reshape(m, -1).T / 255
    Y = np.zeros((m, 10))
    Y[np.arange(m), label] = 1
    Y = Y.T
    return X, Y


X_train, Y_train = process_data(X_train, label_train)
X_test, Y_test = process_data(X_test, label_test)

X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T


#implementing Logistic Regression algorithm
class LogisticRegression() :
    def __init__( self, learning_rate, iterations ) :
        self.learning_rate = learning_rate
        self.iterations = iterations 
        
    def fit( self, X, Y ): 
        self.m, self.n = X.shape 
        self.W = np.matrix(np.zeros( self.n ))
        self.b = 0
        self.X = X 
        self.Y = Y 
        self.c = []
        print(self.X.shape, self.Y.shape, self.W.shape)
    
        for i in range(self.iterations) :
            self.update_weights()
        return self
    
    def update_weights(self): 
        Y_pred = self.predict( self.X )
        cost = log_loss(self.Y, Y_pred)
        dW = - (( self.X.T).dot(self.Y - Y_pred) ) / self.m 
        db = - np.sum( self.Y - Y_pred ) / self.m
        self.W = self.W - self.learning_rate * np.sign(dW.T)
        self.b = self.b - self.learning_rate * np.sign(db)
        self.c.append(cost/self.m)
        return self
    
    def predict(self, X): 
        a = (X.dot(self.W.T)) + self.b
        sigmoid = 1.0/(1 + np.exp(-a)) 
        return sigmoid


#fitting data to Logistic Regression model --- one vs rest
def get_all_weights(data_X, data_Y,  no_of_classes=[0,1,2,3,4,5,6,7,8,9]):
    m = (data_X.shape)[0]
    weights_per_class = []

    for i in no_of_classes:
        model = LogisticRegression( iterations = 1000, learning_rate = 0.008 ) 
        y = np.matrix(np.zeros(m,))
        y = y.T
        for j in range(0, m):
            idx = 0
            for k in range(0,10):
                if(data_Y[j,k] == 1):
                    idx = k
                    break
            if(idx == i):
                y[j] = 1
            else:
                y[j] = 0
        model = model.fit( data_X, y ) 
        weights_per_class.append(model)
    return weights_per_class

# Utility function to predict output
def predict1(model, X): 
    a = (X.dot(model.W.T)) + model.b
    sigmoid = 1.0/(1 + np.exp(-a)) 
    return sigmoid[0,0]


# Function to get predicted class
def get_predicted_class(X):
    no_of_classes = 10
    predicted_class = np.zeros((X.shape[0], no_of_classes))
    for i in range(0, X.shape[0]):
        Y_pred = 0
        index = 0
        for j in range(0, no_of_classes):
            p = predict1( weights_per_class[j], X[i] )
            if(p > Y_pred):
                Y_pred = p
                index = j
        predicted_class[i][index] = 1
    return predicted_class

# Function to find accuracy
def accuracy(pred, actual):
    r, c = pred.shape
    cnt = 0
    for i in range(0, r):
        lst1 = pred[i]
        lst2 = actual[i]
        for j in range(0,10):
            if(lst1[j] != lst2[j]):
                break
        if(j == 9):
            cnt += 1
    acc = cnt/r
    return acc

# getting weights for all class - using one vs rest classification method
weights_per_class = get_all_weights(X_train, Y_train)

x= []
y = []
for i in range(0,1000):
    x.append(i)
    y.append(weights_per_class[0].c)

# Plotting graph of cost vs iteration
plt.plot(x, weights_per_class[0].c) 
plt.xlabel('iteration') 
plt.ylabel('cost') 
plt.title('cost Vs iteration') 
plt.show() 

# Testing trained model on test data
Y_predicted = get_predicted_class(X_test)

# Getting accuracy
ac = accuracy(Y_predicted, Y_test)
print("accuracy:",ac)

