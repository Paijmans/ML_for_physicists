#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mond Dec 30 2019

@author: Joris Paijmans - paijmans@pks.mpg.de
"""
from numpy import *

class SimpleNN:
    """
    Implementation of a simple neural net with back propagation in python.
    """
   
    def __init__(self, LayerSizes=[2,30,30,30,30,1], batchsize = 1000):
        
        self.NumLayers  = len(LayerSizes)-1
        self.LayerSizes = LayerSizes
        self.batchsize  = batchsize
                
        # Initialize neuron weights and biases to random numbers.
        self.reset_weights()
        
        # Set memory for arrays for back propagation.
        self.reset_memory()
        
        self.y_out_result= []
 
    
    def reset_weights(self):
        
        # Initialize neuron weights and biases to random numbers.
        self.Weights = [random.uniform(low=-0.5, high=+0.5,size=[ self.LayerSizes[j], self.LayerSizes[j+1] ]) for j in range(self.NumLayers)]
        self.Biases  = [zeros(self.LayerSizes[j+1]) for j in range(self.NumLayers)]   
        
    def reset_memory(self):
        self.y_layer  = [zeros(self.LayerSizes[j]) for j in range(self.NumLayers+1)]
        self.df_layer = [zeros(self.LayerSizes[j+1]) for j in range(self.NumLayers)]
        self.dw_layer = [zeros([self.LayerSizes[j], self.LayerSizes[j+1]]) for j in range(self.NumLayers)]
        self.db_layer = [zeros(self.LayerSizes[j+1]) for j in range(self.NumLayers)]        
        
        
    #def net_f_df(self, z): # calculate f(z) and f'(z)
    #    val=1/(1+exp(-z))
    #    return(val,exp(-z)*(val**2)) # return both f and f'
    
    # implement a ReLU unit (rectified linear)
    def net_f_df(self, z): # calculate f(z) and f'(z)
        val=z*(z>0)
        return(val,z>0) # return both f and f'    
    
    def forward_step(self, y, w, b): # calculate values in next layer, from input y
        z=dot(y,w)+b # w=weights, b=bias vector for next layer
        return(self.net_f_df(z)) # apply nonlinearity and return result
    
    def apply_net(self, y_in): # one forward pass through the network

        y=y_in # start with input values
        self.y_layer[0]=y
        for j in range(self.NumLayers): # loop through all layers [not counting input]
            # j=0 corresponds to the first layer above the input
            y, df = self.forward_step(y,self.Weights[j],self.Biases[j]) # one step, into layer j
            self.df_layer[j] = df # store f'(z) [needed later in backprop]
            self.y_layer[j+1] = y # store f(z) [also needed in backprop]        
        return(y)

    def apply_net_simple(self, y_in): # one forward pass through the network
        # no storage for backprop (this is used for simple tests)

        y=y_in # start with input values
        for j in range(self.NumLayers): # loop through all layers
            self.y_layer[j]=y
            # j=0 corresponds to the first layer above the input
            y, df = self.forward_step(y, self.Weights[j], self.Biases[j]) # one step
        return(y)    
    
    def backward_step(self, delta, w, df): 
        # delta at layer N, of batchsize x layersize(N))
        # w between N-1 and N [layersize(N-1) x layersize(N) matrix]
        # df = df/dz at layer N-1, of batchsize x layersize(N-1)
        return( dot( delta, transpose(w) )*df )    
    
    def backprop(self, y_target): # one backward pass through the network
        # the result will be the 'dw_layer' matrices that contain
        # the derivatives of the cost function with respect to
        # the corresponding weight

        self.delta = (self.y_layer[-1] - y_target) * self.df_layer[-1]
        self.dw_layer[-1] = dot( transpose( self.y_layer[-2]), self.delta )/self.batchsize
        self.db_layer[-1] = self.delta.sum(0)/self.batchsize
        for j in range(self.NumLayers-1):
            self.delta = self.backward_step(self.delta, self.Weights[-1-j], self.df_layer[-2-j])
            self.dw_layer[-2-j] = dot(transpose(self.y_layer[-3-j]), self.delta)/self.batchsize
            self.db_layer[-2-j] = self.delta.sum(0)/self.batchsize    

    def gradient_step(self, eta): # update weights & biases (after backprop!)

        for j in range(self.NumLayers):
            self.Weights[j] -= eta * self.dw_layer[j]
            self.Biases[j]  -= eta * self.db_layer[j]

    def train_net(self, y_in, y_target, eta): # one full training batch
        # y_in is an array of size batchsize x (input-layer-size)
        # y_target is an array of size batchsize x (output-layer-size)
        # eta is the stepsize for the gradient descent

        self.y_out_result = self.apply_net(y_in)
        self.backprop(y_target)
        self.gradient_step(eta)
        cost = 0.5 * ( (y_target - self.y_out_result)**2 ).sum()/self.batchsize
        return(cost)        