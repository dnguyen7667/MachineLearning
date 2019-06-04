# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:30:41 2019

@author: duong
"""

#To practice building a Linear Regression Model


import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#to create reproducible data
np.random.seed(76)


#generate synthesic data

#input is 100 random data points from uniform distribuition (0,1)

X_original = np.random.rand(10000,1)

#target/output data are supposed to be 4+3X with some random noise generated from standard normal distribuition
y = 4 + 3*X_original - 8*X_original**2 + 15*X_original**3 


#CLOSED-FORM FORMULA: w = inverse(X_t*X)*X_t*y

        
class m_PolynomialRegression():
    """
    To fit polynomial regression into input data.
    Degree of polynomial is to be specified or will be assumed to be 1 - linear.
    If the size of input is > 100,000 then do Gradient Descent instead of Normal Equation.
    
    """
    #theta= regularization term
    
    def __init__(self, order = 1, theta = 0, eta = 0.01):
        """
        @order = order of polynomial, default = 1 (linear)
        @theta = bias-variance trade-off value
        @eta   = learning rate, used in gradient_descent function
        """
        self.order = order
        self.theta = theta
        self.eta = eta 
        
        
        #shape of new, "adjusted" matrix X is (input_size, order+1)
        
        
    #END __init__ function
        
        
    def normal_equation(self):
        #To use if there are less than 100,000 data points
        
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X_new_t, self.X_new)), self.X_new_t), self.y)
        
        self.intercept = self.w[0]
        self.coefs = self.w[1:]
     #END normal_equation function   
        
     
    def gradient_descent(self ):
        #To use if there are more than 100,000 data points
        #stopping criteria = when there's no significant changes in cost function, or
        #               gradient is 0, 
        #               or after a specified number of steps (500 steps this case), which ever comes first
        
        #weight vector has dimension of (order+1) by 1
        
        weights = np.random.randn(self.order+1, 1)
        steps= 500
        #weights_next = np.zeros((self.order+1, 1))
        
                
        for step in range(steps):
            gradients = 2/self.num_data_points * self.X_new_t.dot(self.X_new.dot(weights) - self.y)
            
            weights = weights - self.eta*gradients
    
        print ("Weights from gradient descent function : ", weights)
        
        
    #end gradien_descent_function 
   
        
    def fit(self, X, y):
        self.X_new = np.zeros(shape= (X.shape[0], self.order+1))
        
        #create new columns of attributes as a second, third, forth... order
        #then treat it like a regular linear regression with multiple attributes
        for i in range(self.order+1):
            self.X_new[:,i] = X[:,0]**i
        
        self.X_new_t = self.X_new.T
        
        self.num_data_points = X.shape[0]
        self.y = y
        
        if (self.num_data_points < 100000):
            self.normal_equation()
        else:
            self.gradient_descent()
        
        
    def predict(self, X):
        self.X_new = np.c_[np.ones(X.shape[0]), X]
        self.X_new_t = self.X_new.transpose()
        
        self.predictions = np.matmul(self.w.transpose(), X_new_t).transpose()
        
        print(self.predictions)
        return self.predictions
            
            
        
        
        
        
        
        
        
        
    
