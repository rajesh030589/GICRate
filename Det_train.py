# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 03:49:07 2018

@author: rajes
"""
import numpy as np
import random as rn


def Symbol_mapper(x):
    return {
            '00': .707 + 1j*.707,
            '01': .707 - 1j*.707,
            '11': -.707 - 1j*.707,
            '10': -.707 + 1j*.707,
            
            }[x]
        
n = 1
R = 2

N0 = np.sqrt(.25)  # Noise Power
#Iter = 100000
Iter = 1000
S = 0
XX = np.zeros([Iter,3])
for i in range(Iter):
    # Choose integers from 0 to 2^(nR) - 1
    Xi = rn.randint(0,2**(n*R) - 1)
    
    # Convert the integers to bits
    X  = '{0:02b}'.format(Xi)
    Xs = Symbol_mapper(X)
    z  = rn.gauss(0,N0/2) + 1j* rn.gauss(0,N0/2)
    
    Ys = Xs + z;
    
    XX[i,0] = np.real(Ys)
    XX[i,1] = np.imag(Ys)
    XX[i,2] = Xi
    
#np.savetxt('Train.csv',XX,delimiter = ',')    
np.savetxt('Test.csv',XX,delimiter = ',')    
    
