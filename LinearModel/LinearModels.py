import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import random

#START Function make_X_b
def make_X_b(inputs):
    """ This function is to "add 1 to the front of each training instances then return the new matrix. """
    #if X is data frame
    #then just get
    mInputs= inputs.copy()
    if (type(mInputs)== pd.DataFrame):
        X= mInputs.values #get values in a np array form
            #self.num_attributes= self.X.shape[1] #get number of attributes
    else: #if inputs is already np array
        X= mInputs
            
    return np.c_[np.ones(shape= X.shape[0]), X] #add 1 to the begining of each element of X
#END Function make_X_b


#START Class mLinearRegression
class mLinearRegression:
    """A simple Linear Regression class that will:
    Return predictions used Normal-Equation(closed form) of Linear Regression algorithm.
    Advantages: Easy to implement, return global optimal, predictions are calculated fast once model is trained.
    Disadvantages: Very slow when number of features get too large and or too many training instances (cannot fit all in memory)"""
        
    def __init__(self):
        pass
    
    def fit(self, inputs, output):
        self.X_b= make_X_b(inputs) 
        
        #to make sure that if output is a pd Series, it would be converted to np array
        self.y= np.array(output).copy().reshape(-1,1)
        self.theta_hat= np.linalg.inv(self.X_b.T.dot(self.X_b)).dot(self.X_b.T).dot(self.y) #optimal theta_hat according to the equation
      
    def predict(self, inputs):
        #transform means we use inputs and self.theta_hat to return predicted values of outputs(or y's)
        #return array of predictions
        #y_hat= theta_hat.T dot X (inputs)
        
        self.inputs_b= make_X_b(inputs)
        return self.theta_hat.T.dot(self.inputs_b.T)
    
    #to get optimal theta hat(weights)
    def get_params(self):
        return self.theta_hat     
    
#END Class mLinearRegression    
    
    
    
    
#The idea of Gradient Descent is to tweak parameters iteratively in order to minimize cost function.
        
#START Class mBatchGD
class mBatchGD:
    """Batch Gradient Descent.
    To minimize cost function= compute gradient of the cost function
    with regards to each model parameter theta_sub_j.
    Initially theta_hat(weights) are generated randomly."""
    
    #Need learning rate (eta), large #s of epochs (willing to wait long)
    
    def __init__(self, random_state= None, eta= 0.1, max_iter= 1000):
        if(random_state!= None):
            random.seed(random_state)
        self.eta= eta
        self.max_iter= max_iter
            
       
    def fit(self, inputs, output):
        self.X_b= make_X_b(inputs)
        self.y= np.array(output).copy().reshape(-1,1)
        
        self.num_instances= len(self.y) #number of training instances
        self.num_dim= self.X_b.shape[1] #number of dimensions(attributes)
        
        self.X= self.X_b.T
        
        #randomly initialize
        self.theta_hat= np.random.rand(self.num_dim).reshape(-1,1)
        
        for i in range(self.max_iter):
            del_mse= 2*self.X.dot(self.X_b.dot(self.theta_hat) - self.y)/self.num_instances
            self.theta_hat= self.theta_hat - self.eta*del_mse
           
    """
    Args:
        None
    Returns:
        Best thetas (weights) 
    """         
    def get_params(self):
        return self.theta_hat
    
#END Class mBatchGD        
        

        
            
