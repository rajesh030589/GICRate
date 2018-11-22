# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 05:52:26 2018

@author: rajes

Desription: This implements a basic SISO link to transmit a QPSK symbol with 
R = 2 and n = 1 and record the bit error rate

"""

import numpy as np
import random as rn
import matplotlib.pyplot as plt


def Symbol_mapper(x):
    return {
            '00': .707 + 1j*.707,
            '01': .707 - 1j*.707,
            '11': -.707 - 1j*.707,
            '10': -.707 + 1j*.707,
            
            }[x]

def Detector(x):
    if(np.real(x)>0):
        if(np.imag(x)>0):
            return '00'
        else:
            return '01'
    else:
        if(np.imag(x)>0):
            return '10'
        else:
            return '11'
        
n = 1
R = 2

N0 = np.sqrt(.25)  # Noise Power
Iter = 100000

S = 0
for i in range(Iter):
    # Choose integers from 0 to 2^(nR) - 1
    X = rn.randint(0,2**(n*R) - 1)
    
    # Convert the integers to bits
    X  = '{0:02b}'.format(X)
    Xs = Symbol_mapper(X)
    z  = rn.gauss(0,N0/2) + 1j* rn.gauss(0,N0/2)
    
    Ys = Xs + z;
    
    # Received Signal Constellation
    plt.scatter(np.real(Ys),np.imag(Ys))
    
    # Detection
    Y = Detector(Ys)
    
    if(Y[0] != X[0] and Y[1] != X[1]):
        S = S + 2;
    elif(Y[0] != X[0]):
        S = S + 1
    elif(Y[1] != X[1]):
        S = S + 1;
        
BER = S/Iter

print(BER)
